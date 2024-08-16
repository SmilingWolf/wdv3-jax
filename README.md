# wdv3-jax

Small example thing showing how to use JAX/FLAX to run the WD Tagger V3 models.

Base code ~~shamelessly stolen~~ borrowed from https://github.com/neggles/wdv3-timm

The Models directory has been copied from https://github.com/SmilingWolf/JAX-CV  
One day I might actually package my code like a normal person, but until then...

## How To Use

1. clone the repository and enter the directory:
```sh
git clone https://github.com/SmilingWolf/wdv3-jax.git
cd wdv3-jax
```

2. Create a virtual environment and install the Python requirements.

If you're using Linux, you can use the provided script:
```sh
bash setup.sh
```

Or if you're on Windows (or just want to do it manually), you can do the following:
```sh
# Create virtual environment
python3.11 -m venv .venv
# Activate it
source .venv/bin/activate
# Upgrade pip/setuptools/wheel
python -m pip install -U pip setuptools wheel
# At this point, optionally you can install JAX manually (e.g. if you are using an nVidia GPU)
python -m pip install -U "jax[cpu]"
# Install requirements
python -m pip install -r requirements.txt
```

Consult https://github.com/google/jax?tab=readme-ov-file#installation for more infos on how to install JAX with GPU/TPU/ROCm/Metal support

3. Run the example script, picking one of the 3 models to use:
```sh
python wdv3_jax.py --model <swinv2|convnext|vit> path/to/image.png
```

Example output from `python wdv3_jax.py --model swinv2 test.png`:
```sh
Loading model 'swinv2' from 'SmilingWolf/wd-swinv2-tagger-v3'...
Loading tag list...
Loading image and preprocessing...
Running inference...
Processing results...
--------
Caption: 1girl, whiskey, eyepatch, playboy_bunny, animal_ears, pantyhose, rabbit_ears, solo, breasts, alcohol, braid, long_hair, leotard, scar, multicolored_hair, wrist_cuffs, black_pantyhose, holding, scar_on_face, fake_animal_ears, covered_navel, bottle, looking_at_viewer, grey_hair, sky, medium_breasts, star_(sky), starry_sky, night, black_leotard, yellow_eyes, streaked_hair, strapless_leotard, cleavage, strapless, long_braid, detached_collar, night_sky, cup, bare_shoulders, single_braid, blush, scar_across_eye, holding_bottle, standing, m16a1_(girls'_frontline)
--------
Tags: 1girl, whiskey, eyepatch, playboy bunny, animal ears, pantyhose, rabbit ears, solo, breasts, alcohol, braid, long hair, leotard, scar, multicolored hair, wrist cuffs, black pantyhose, holding, scar on face, fake animal ears, covered navel, bottle, looking at viewer, grey hair, sky, medium breasts, star \(sky\), starry sky, night, black leotard, yellow eyes, streaked hair, strapless leotard, cleavage, strapless, long braid, detached collar, night sky, cup, bare shoulders, single braid, blush, scar across eye, holding bottle, standing, m16a1 \(girls' frontline\)
--------
Ratings:
  general: 0.003
  sensitive: 0.986
  questionable: 0.012
  explicit: 0.000
--------
Character tags (threshold=0.75):
  m16a1_(girls'_frontline): 0.999
--------
General tags (threshold=0.35):
  1girl: 0.999
  whiskey: 0.996
  eyepatch: 0.983
  playboy_bunny: 0.979
  animal_ears: 0.978
  pantyhose: 0.975
  rabbit_ears: 0.974
  solo: 0.952
  breasts: 0.948
  alcohol: 0.913
  braid: 0.888
  long_hair: 0.885
  leotard: 0.881
  scar: 0.864
  multicolored_hair: 0.860
  wrist_cuffs: 0.850
  black_pantyhose: 0.829
  holding: 0.801
  scar_on_face: 0.792
  fake_animal_ears: 0.754
  covered_navel: 0.740
  bottle: 0.739
  looking_at_viewer: 0.737
  grey_hair: 0.672
  sky: 0.670
  medium_breasts: 0.669
  star_(sky): 0.659
  starry_sky: 0.644
  night: 0.643
  black_leotard: 0.642
  yellow_eyes: 0.635
  streaked_hair: 0.633
  strapless_leotard: 0.605
  cleavage: 0.601
  strapless: 0.576
  long_braid: 0.546
  detached_collar: 0.523
  night_sky: 0.522
  cup: 0.490
  bare_shoulders: 0.467
  single_braid: 0.444
  blush: 0.428
  scar_across_eye: 0.424
  holding_bottle: 0.421
  standing: 0.413
Done!
```
