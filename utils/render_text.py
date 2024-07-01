from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont


def parse_yaml(yaml_file, shape, font):
    config = OmegaConf.load(yaml_file).Config
    texts = config.texts
    top_left_x = config.top_left_x
    top_left_y = config.top_left_y
    width = config.width
    height = config.height

    assert len(texts) == len(top_left_x) and len(texts) == len(top_left_y) and \
        len(texts) == len(width) and len(texts) == len(height)

    render_list = []
    for i in range(len(texts)):
        text_dict = {}
        text_dict['text'] = texts[i]
        text_dict['font_name'] = font
        text_dict['polygon'] = list(
            map(
                int, 
                [
                    shape[0] * top_left_x[i],
                    shape[1] * top_left_y[i],
                    shape[0] * width[i],
                    shape[1] * height[i]
                ]
            )
        )
        render_list.append(text_dict)
    
    return render_list
    

def render_all_text(render_list, shape):
    width = shape[0]
    height = shape[1]
    board = Image.new('RGB', (width, height), 'black')

    for text_dict in render_list:
        text = text_dict['text']
        polygon = text_dict['polygon']
        font_name = text_dict['font_name']

        w, h = polygon[2:]
        vert = True if w < h else False
        image4ratio = Image.new('RGB', (1024, 1024), 'black')
        draw = ImageDraw.Draw(image4ratio)

        try:
            font = ImageFont.truetype(f'./fonts/{font_name}.ttf', encoding='utf-8', size=50)
        except FileNotFoundError:
            font = ImageFont.truetype(f'./fonts/{font_name}.otf', encoding='utf-8', size=50)

        if not vert:
            draw.text(xy=(0, 0), text=text, font=font, fill='white')
            _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
            _th += 1
        else:
            _tw, y_c = 0, 0
            for c in text:
                draw.text(xy=(0, y_c), text=c, font=font, fill='white')
                _l, _t, _r, _b = font.getbbox(c)
                _tw = max(_tw, _r - _l)
                y_c += _b
            _th = y_c + 1

        ratio = (_th * w) / (_tw * h)
        text_img = image4ratio.crop((0, 0, _tw, _th))
        x_offset, y_offset = 0, 0
        if 0.8 <= ratio <= 1.2:
            text_img = text_img.resize((w, h))
        elif ratio < 0.75:
            resize_h = int(_th * (w / _tw))
            text_img = text_img.resize((w, resize_h))
            y_offset = (h - resize_h) // 2
        else:
            resize_w = int(_tw * (h / _th))
            text_img = text_img.resize((resize_w, h))
            x_offset = (w - resize_w) // 2

        board.paste(text_img, (polygon[0] + x_offset, polygon[1] + y_offset))

    return board
