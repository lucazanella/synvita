import os
from .clip_encoder import CLIPVisionTower
from .languagebind import LanguageBindImageTower, LanguageBindVideoTower

# ============================================================================================================

def build_image_tower(image_tower_cfg, **kwargs):
    # image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    image_tower = 'cache_dir/models--LanguageBind--LanguageBind_Image/snapshots/d8c2e37b439f4fc47c649dc8b90cdcd3a4e0c80e'
    # is_absolute_path_exists = os.path.exists(image_tower)
    # if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
    #     return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    # if image_tower.endswith('LanguageBind_Image'):
    #     return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    # video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    video_tower = 'cache_dir/models--LanguageBind--LanguageBind_Video_merge/snapshots/efc40ec6ba6b2081276c11e7e19b24f08a099e79'
    # if video_tower.endswith('LanguageBind_Video_merge'):
    #     return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    # raise ValueError(f'Unknown video tower: {video_tower}')
    return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
# ============================================================================================================
