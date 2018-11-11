import random
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont


def randomChar():
    return chr(random.randint(48, 57))


def randomBgColor():
    return (random.randint(50,100), random.randint(50, 100), random.randint(50, 100))


def randomTextColor():
    return (random.randint(120, 200), random.randint(120, 200), random.randint(120, 200))


w = 30 * 4
h = 60

font = ImageFont.truetype(font=r'E:\Pycharmprojects\SEQ2SEQ\arial.ttf', size=40)

for _ in range(5000):

    image = Image.new('RGB', (w, h), (255, 255, 255))

    draw = ImageDraw.Draw(image)

    for x in range(w):
        for y in range(h):
            draw.point((x, y), fill=randomBgColor())

    filename = []
    for t in range(4):
        ch = randomChar()
        filename.append(ch)
        draw.text((30 * t, 10), ch, font=font, fill=randomTextColor())

    image.save('./Code/{0}.jpg'.format(''.join(filename)), 'Jpeg')