# generate_order.py
python3 src/generator/generate_order.py
**Check newest order**
ls -lt data/orders
# render_captchas.php
Captcha generator using Gregwar Captcha library
Renders captcha images to order and writes labels to a CSV file.

**CMD**
php src/generator/render_captchas.php data/orders/order_000X 5Char_100000_CapGen_balanced
Replace:
	•	order_000X → latest folder from above
	•	second argument → your dataset name

Current phrase: php src/generator/render_captchas.php data/orders/order_0006 5Char_Alphabet_CapGen 