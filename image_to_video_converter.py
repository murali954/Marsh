import cv2
import os


image_dir = r"C:\Users\sree\Desktop\flowers"

# Output video file
output_video_path = r"C:\Users\sree\Desktop\output_video.avi"

# Get all image files in the directory
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
image_files.sort()


width, height = 640, 480

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 1, (width, height))


for image_file in image_files:
    image = cv2.imread(image_file)

    # Resize image to match the specified dimensions
    image = cv2.resize(image, (width, height))

    out.write(image)


out.release()

print(f"Video created successfully! Path: {output_video_path}")
