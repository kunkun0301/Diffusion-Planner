import json
import torch

from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer


class Config:
    
    def __init__(
            self,
            args_file,
            guidance_fn
    ):
        with open(args_file, 'r') as f:
            args_dict = json.load(f)
            
        for key, value in args_dict.items():
            setattr(self, key, value)

        # 默认短期预测长度：如果配置里没有，就用 future_len 的一半
        if not hasattr(self, "short_future_len"):
            assert hasattr(self, "future_len"), "future_len must be set in args_file when short_future_len is not provided."
            self.short_future_len = self.future_len // 2

        # 默认不开启 LongShortDecoder，除非在配置里显式打开
        if not hasattr(self, "use_longshort_decoder"):
            self.use_longshort_decoder = False

        self.state_normalizer = StateNormalizer(self.state_normalizer['mean'], self.state_normalizer['std'])
        self.observation_normalizer = ObservationNormalizer({
            k: {
                'mean': torch.as_tensor(v['mean']),
                'std': torch.as_tensor(v['std'])
            } for k, v in self.observation_normalizer.items()
        })
        
        self.guidance_fn = guidance_fn
