import numpy as np

f = np.load('/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x768_fixed.npz', allow_pickle=True)
sids = f['study_ids']
N = len(sids)
print(f'{N} clips')

rng = np.random.default_rng(42)
embs = rng.standard_normal((N, 768)).astype(np.float16)

np.savez('/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/random_clips_4x768.npz',
         embeddings=embs, study_ids=sids)
print('done')
