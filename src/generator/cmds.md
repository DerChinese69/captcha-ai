# generate_order.py
Generates a new order manifest for captcha rendering.

`classes`, `length`, and `target_per_class_per_position` must be passed explicitly.

**CMD**
```bash
python3 src/generator/generate_order.py \
  --classes "ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
  --length 5 \
  --target-per-class-per-position 1000
```

Replace:
- `--classes` with the exact character set you want to generate
- `--length` with the captcha label length
- `--target-per-class-per-position` with the target count for each class at each position

**Check newest order**
```bash
ls -lt data/orders
```

# render_captchas.php
Captcha generator using Gregwar Captcha library
Renders captcha images to order and writes labels to a CSV file.

**CMD**
```bash
php src/generator/render_captchas.php data/orders/order_000X 5Char_100000_CapGen_balanced
```
Replace:
- `order_000X` with the latest folder from above
- the second argument with your dataset name

Current phrase:
python3 src/generator/generate_order.py \
  --classes "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" \
  --length 5 \
  --target-per-class-per-position 1000

  php src/generator/render_captchas.php data/orders/order_0009 5Char_36k_AlpNum
  php src/generator/render_captchas.php data/orders/order_0010 5Char_360k_AlpNum 