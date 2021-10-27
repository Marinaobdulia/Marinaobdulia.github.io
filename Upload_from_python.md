# Instagram picture uploader
---

The aim of this project was to upload pictures from my laptop to my instagram account programatically. Prior to uploading the pictures, they needed to be reformatted (as they were taken from an iPhone) and resized to meet instagram requirements.

Also, since the pictures were from inktober illustrations, captions were created based on the prompt list using a template.


```python
from instabot import Bot
from wand.image import Image
import os
from PIL import Image, ImageOps
```


```python
# Change image format
SourceFolder="C:/Users/merii/Desktop/inktober/"
TargetFolder="C:/Users/merii/Desktop/inktober/"

for file in os.listdir(SourceFolder):
    SourceFile=SourceFolder + "/" + file
    TargetFile=TargetFolder + "/" + file.replace(".HEIC",".JPG")

    img=Image(filename=SourceFile)
    img.format='jpg'
    img.save(filename=TargetFile)
    img.close()
```


```python
# Resize images
for i in range(len(my_captions)):
    pic = 'C:/Users/merii/Desktop/inktober/'+str(i+9) + '.jpg'
    original_image = Image.open(pic) 
    img = ImageOps.exif_transpose(original_image) #to avoid the rotated pictures from rotating again
    img = img.resize((1080,1350), Image.ANTIALIAS)
    img.save('C:/Users/merii/Desktop/inktober/resize_'+str(i+9) + '.jpg')
```


```python
# Initialise instabot
bot = Bot()
bot.login(username = "outofcomfort",
          password = "********")
```
Watch.

#watercolor #acuarela #watercolorart #artoftheday #acuarelle 
#art #tinytattoo #nature #free #doodle #inktober #inktober8 #inktober2021 #intoberwatch

```python
# Create captions based on inktober prompt list.
inktober = ['pressure','pick','sour','stuck',
            'roof','tick','helmet','compass',
            'collide','moon','loop','sprout',
            'fuzzy','open','leak','extinct',
            'splat','connect','spark','crispy',
            'patch','slither','risk']

my_captions = []
counter = 8
for prompt in inktober:
    counter +=1
    my_captions.append(prompt.capitalize()+ f""".\n 
    #watercolor #acuarela #watercolorart #artoftheday  
    #acuarelle #art #tinytattoo #nature #free #doodle 
    #inktober #inktober{str(counter)} #inktober2021 #intober{prompt}
    #uploadedfrompython""".format())
```


```python
# Upload images
for i in range(len(my_captions)):
    pic = 'C:/Users/merii/Desktop/inktober/resize_'+str(i+9) + '.jpg'
    bot.upload_photo(pic, caption = my_captions[i])
```
