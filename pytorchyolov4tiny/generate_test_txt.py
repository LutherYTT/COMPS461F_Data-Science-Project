import os

image_files = []
#os.chdir(os.path.join("/content/gdrive/MyDrive/FYP/output/images/train"))
for filename in os.listdir("/content/images/test"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_files.append("/content/images/test/" + filename)

with open("test.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
