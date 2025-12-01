# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Callable, Iterable, List, Optional, Sequence

import torch

from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Subset

from torch.utils.data.distributed import DistributedSampler

"""
把多个 PyTorch DataLoader 按给定概率“混合”成一个迭代器，每次迭代从某个子 DataLoader 抽一批数据（按混合概率随机选择子 DataLoader）。
"""
class MixedDataLoader:
    def __init__(self, dataloaders: List[DataLoader], mixing_prob: torch.FloatTensor):
        """
        Args:
            dataloaders (List[DataLoader]): List of DataLoaders to be mixed.
            mixing_prob (torch.FloatTensor): Probability of each dataloader to be sampled from

        """
        assert len(dataloaders) == mixing_prob.shape[0]
        self.dataloaders = dataloaders
        self.mixing_prob = mixing_prob
        # Iterator state
        self._iter_dls = None
        self._iter_mixing_prob = None
        self.random_generator = torch.Generator()
        """
        class MixedDataLoader: 定义了一个新类，名字是 MixedDataLoader。
        __init__ 是构造函数（对象被创建时会自动运行）。函数参数使用了类型注解：
        dataloaders: List[DataLoader]：期望是一个 DataLoader 的列表。
        mixing_prob: torch.FloatTensor：一个 PyTorch tensor，描述每个 dataloader 被选择的概率（例如 [0.15, 0.7, 0.15]）。
        assert len(dataloaders) == mixing_prob.shape[0]：这个断言确保 dataloaders 数量和概率向量长度一致，
        否则抛错。assert 用于写“假设条件”，条件不满足程序会报 AssertionError 并终止（常用于开发阶段保证前提）。
        self._iter_dls、self._iter_mixing_prob：类里用来存储迭代器状态的变量，初始设为 None。把它们设为 None 意味着“还没开始迭代”。
        self.random_generator = torch.Generator()：PyTorch 的随机数生成器对象。跟 Python 自带的 random 或 numpy.random 类似，
        但这是 torch 级的随机生成器，可以用于 torch.multinomial 等。用这个可以控制随机种子，从而保证可复现性（每次训练如果固定种子就能得到相同的抽样序列）。
        """

    def __len__(self):
        return sum([len(d) for d in self.dataloaders])

    def __iter__(self):
        # Synchronize dataloader seeds
        self.random_generator.manual_seed(42)
        self._iter_dls = [iter(loader) for loader in self.dataloaders]
        self._iter_mixing_prob = self.mixing_prob.clone()
        return self
    """
    self.random_generator.manual_seed(42)：
    为随机生成器设定种子 42，保证每次调用 iter(mixed_loader) 时会从相同的随机状态开始（方便复现）。
    注意：这样每次 __iter__ 都设同样种子，会导致相同序列；这在某些训练场景下是需要的（能让数据切分一致），但也可能在多进程训练时需要更复杂的 seed 管理。
    self._iter_dls = [iter(loader) for loader in self.dataloaders]：
    对每个子 DataLoader 调用 iter(...)，得到其迭代器，这样 next(...) 就可以从每个 dataloader 取 batch。
    self._iter_mixing_prob = self.mixing_prob.clone()：
    复制一份概率张量。注意要 .clone()，这样接下来如果修改 _iter_mixing_prob（例如把某个 dataset 的概率置 0），不会改变原来的 self.mixing_prob。
    return self：把自己作为迭代器对象返回（这是常见模式）。
    """

    def __next__(self):
        """
        Sample a dataloader to sample from based on mixing probabilities. If one of the dataloaders is exhausted, we continue sampling from the other loaders until all are exhausted.
        """
        if self._iter_dls is None:
            raise TypeError(f"{type(self).__name__} object is not an iterator")

        while self._iter_mixing_prob.any():  # at least one D-Loader with non-zero prob.
            dataset_idx = self._iter_mixing_prob.multinomial(
                1, generator=self.random_generator
            ).item()
            try:
                item = next(self._iter_dls[dataset_idx])
                return item
            except StopIteration:
                # No more iterations for this dataset, set it's mixing probability to zero and try again.
                self._iter_mixing_prob[dataset_idx] = 0
            except Exception as e:
                # log and raise any other unexpected error.
                logging.error(e)
                raise e
        """
        if self._iter_dls is None:：
        如果没有调用 __iter__（即 self._iter_dls 还是 None），表明对象还不是迭代状态，按习惯抛出 TypeError
        （这是 Python 里常见的做法：在没有被初始化为迭代器时调用 next 应报错）。

        while self._iter_mixing_prob.any():：
        只要 _iter_mixing_prob 中还有非零元素（即至少有一个 dataloader 仍可选），
        就持续尝试从某个 dataloader 取数据。
        
        dataset_idx = self._iter_mixing_prob.multinomial(1, generator=self.random_generator).item()：
        这条很重要。tensor.multinomial(n) 意味着从 tensor 的元素当作概率分布抽样 n 个索引（不放回）。
        这里 n=1，所以返回一个长度为 1 的张量，.item() 把它变成普通 Python int。

        generator=self.random_generator：使用上面我们设置并固定种子的随机生成器，保证可复现。
        结果 dataset_idx 就是用来选择哪一个子 dataloader 的索引。
        
        try: item = next(self._iter_dls[dataset_idx]); return item：
        对选中的子 dataloader 的迭代器调用 next，如果成功就返回 item —— 这个 item 通常就是一个 batch（例如图片张量、标签等）。

        except StopIteration:：如果该子 dataloader 已经迭代完（没有更多 batch），next 会抛 StopIteration。这段捕获后：
        把 _iter_mixing_prob[dataset_idx] = 0：把该 dataloader 的采样概率设为 0，表示以后不再选它（因为它已空）。

        然后 while 循环继续，去选其他 dataloader。

        except Exception as e:：捕获到其它任意异常，先 logging.error(e) 记录日志，再 raise e 抛出，方便上层看到问题并停掉训练。

        当 while 条件不成立（所有 _iter_mixing_prob 元素都为 0），
        说明所有 dataloader 都耗尽，执行 raise StopIteration 表示整个 MixedDataLoader 也迭代结束 —— 这使得 for batch in mixed_loader: 可以正确地在末尾停掉。
        """
        # Exhausted all iterators
        raise StopIteration
        """
        这个类把多个 DataLoader 当成几堆牌，
        当你每次要抽一手牌时，会根据概率决定从哪堆抽一手；如果某堆牌抽完，它就被移除（概率设为 0），直到所有堆都抽完为止。
        """

"""
这是一个高层封装，接收多个 Dataset（以及每个的 batch size 等），为每个 dataset 构造 DataLoader（含分布式采样、batch sampler 等），
并返回一个 MixedDataLoader，从而实现按比例混合训练数据源（比如把图像数据和视频数据混合训练）。
"""
class TorchTrainMixedDataset:
    def __init__(
        self,
        datasets: List[Dataset],    # Dataset 对象列表（每个 dataset 代表一种数据源）。
        batch_sizes: List[int],     # 对应每个 dataset 的 batch size（注意每个 dataset 可以用不同 batch size）。
        num_workers: int,       # 每个 DataLoader 的子进程 worker 数（用于并行加载数据）。
        shuffle: bool,      # 是否打乱数据（用于 DistributedSampler）。
        pin_memory: bool,       # DataLoader 的 pin_memory 参数（如果 True，GPU 拷贝更快）。
        drop_last: bool,        # 是否丢弃最后一个不满 batch 的小批次（训练中常设置 True 以保证批次大小一致）。
        collate_fn: Optional[Callable] = None,      # 如何把若干样本合并成一个 batch 的函数（可选）。
        worker_init_fn: Optional[Callable] = None,      # worker 进程初始化函数（可选）。
        phases_per_epoch: int = 1,      # 将每个 dataset 切分为若干“phase”的数量，便于在一个训练 epoch 中分块遍历大型 dataset（下面代码会解释）。
        dataset_prob: Optional[List[float]] = None,     # 可选的概率数组，指定每个 dataset 被选中的概率。如果不提供，代码会按照 dataset 的“长度”自动计算概率（见后文）。
    ) -> None:
        """
        Args:
            datasets (List[Dataset]): List of Datasets to be mixed.
            batch_sizes (List[int]): Batch sizes for each dataset in the list.
            num_workers (int): Number of workers per dataloader.
            shuffle (bool): Whether or not to shuffle data.
            pin_memory (bool): If True, use pinned memory when loading tensors from disk.
            drop_last (bool): Whether or not to drop the last batch of data.
            collate_fn (Callable): Function to merge a list of samples into a mini-batch.
            worker_init_fn (Callable): Function to init each dataloader worker.
            phases_per_epoch (int): Number of phases per epoch.
            dataset_prob (List[float]): Probability of choosing the dataloader to sample from. Should sum to 1.0
        """

        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        assert len(self.datasets) > 0


        for dataset in self.datasets:
            assert not isinstance(dataset, IterableDataset), "Not supported"
            # `RepeatFactorWrapper` requires calling set_epoch first to get its length
            self._set_dataset_epoch(dataset, 0)
        """
        它确保传入的 dataset 不是 IterableDataset（因为这个封装不支持流式 dataset）。
        IterableDataset 的迭代规则与普通 map-style Dataset（支持 len 和索引）不同，混合处理时会更复杂，所以这个实现直接禁用了它。

        self._set_dataset_epoch(dataset, 0)：
        对 dataset 调用一个函数（下面定义），
        目的是如果 dataset 有 epoch 或 set_epoch 方法就设置为 0（一些 wrapper 如 RepeatFactorWrapper 需要知道 epoch 来决定索引或权重）。
        """

        self.phases_per_epoch = phases_per_epoch
        self.chunks = [None] * len(datasets)
        """
        phases_per_epoch：如果 >1，意味着我们会把每个 dataset 的索引按 phases_per_epoch 切成若干个 chunk（分块），每个大 epoch 会只用其中一块来训练。
        这样可以把非常大的 dataset 分期使用，或者用在抢占/恢复后能从某个 phase 继续。
        
        self.chunks：用于存放每个 dataset 被切分后的索引块列表（每个元素会是 torch.chunk(...) 的结果）。
        """

        if dataset_prob is None:
            # If not provided, assign each dataset a probability proportional to its length.
            dataset_lens = [
                (math.floor(len(d) / bs) if drop_last else math.ceil(len(d) / bs))
                for d, bs in zip(datasets, batch_sizes)
            ]
            total_len = sum(dataset_lens)
            dataset_prob = torch.tensor([d_len / total_len for d_len in dataset_lens])
        else:
            assert len(dataset_prob) == len(datasets)
            dataset_prob = torch.tensor(dataset_prob)

        logging.info(f"Dataset mixing probabilities: {dataset_prob.tolist()}")
        assert dataset_prob.sum().item() == 1.0, "Probabilities should sum to 1.0"
        self.dataset_prob = dataset_prob

    def _set_dataset_epoch(self, dataset, epoch: int) -> None:
        if hasattr(dataset, "epoch"):
            dataset.epoch = epoch
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def get_loader(self, epoch) -> Iterable:
        dataloaders = []
        for d_idx, (dataset, batch_size) in enumerate(
            zip(self.datasets, self.batch_sizes)
        ):
            if self.phases_per_epoch > 1:
                # Major epoch that looops over entire dataset
                # len(main_epoch) == phases_per_epoch * len(epoch)
                main_epoch = epoch // self.phases_per_epoch

                # Phase with in the main epoch
                local_phase = epoch % self.phases_per_epoch

                # Start of new data-epoch or job is resumed after preemtion.
                if local_phase == 0 or self.chunks[d_idx] is None:
                    # set seed for dataset epoch
                    # If using RepeatFactorWrapper, this step currectly re-samples indices before chunking.
                    self._set_dataset_epoch(dataset, main_epoch)

                    # Separate random generator for subset sampling
                    g = torch.Generator()
                    g.manual_seed(main_epoch)
                    self.chunks[d_idx] = torch.chunk(
                        torch.randperm(len(dataset), generator=g),
                        self.phases_per_epoch,
                    )

                dataset = Subset(dataset, self.chunks[d_idx][local_phase])
            else:
                self._set_dataset_epoch(dataset, epoch)

            sampler = DistributedSampler(dataset, shuffle=self.shuffle)
            sampler.set_epoch(epoch)

            batch_sampler = BatchSampler(sampler, batch_size, drop_last=self.drop_last)
            dataloaders.append(
                DataLoader(
                    dataset,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    batch_sampler=batch_sampler,
                    collate_fn=self.collate_fn,
                    worker_init_fn=self.worker_init_fn,
                )
            )
        return MixedDataLoader(dataloaders, self.dataset_prob)
