import kagglehub

# Download latest version
path = kagglehub.dataset_download(r"C:/Users/Akash/Desktop/hackathon/indian-bovine-breeds")
# import os
# path = kagglehub.dataset_download(
#     os.path.join(r"C:/Users/Akash/Desktop/hackathon")
# )


print("Path to dataset files:", path)