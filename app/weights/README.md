# Model Weight Files

Put these files in this folder before starting the server:

| File                    | Comes from                                              |
|-------------------------|---------------------------------------------------------|
| `classifier.pth`        | your `best_weather_model.pth` (renamed)                 |
| `rain_enhancer.pth`     | `checkpoints/rain_pix2pix/latest_net_G.pth`             |
| `fog_enhancer.pth`      | `checkpoints/fog2clear/latest_net_G.pth`                |
| `lowlight_enhancer.pth` | `checkpoints/dark_pix2pix/latest_net_G.pth`             |

Copy only the `*_net_G.pth` (generator) files, never `*_net_D.pth`.

Per-model settings (norm type, resolution) live in `ENHANCERS` in `app/config.py`
and must match each experiment's `train_opt.txt`.

Check everything before starting the server (run from `backend/`):

    python scripts/check_weights.py [test_image.jpg]
