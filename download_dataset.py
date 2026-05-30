import os
import zipfile
import gdown

def main():
    file_id = '1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'VisDrone2019-MOT-val.zip'
    
    print("Downloading VisDrone MOT Validation dataset from Google Drive...")
    if not os.path.exists(output):
        try:
            gdown.download(url, output, quiet=False)
        except Exception as e:
            print(f"Error downloading with gdown: {e}")
            print("Trying fallback curl command...")
            os.system(f'curl -L -o {output} "https://drive.google.com/uc?export=download&confirm=t&id={file_id}"')
    else:
        print("Zip file already exists. Skipping download.")
        
    if os.path.exists(output) and os.path.getsize(output) > 1000000:
        print("Extracting dataset...")
        os.makedirs('data', exist_ok=True)
        try:
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall('data')
            print("Dataset download and extraction complete!")
            # Optionally clean up the zip file to save space
            # os.remove(output)
        except Exception as e:
            print(f"Extraction failed: {e}")
    else:
        print("Downloaded file is invalid or too small. Please check internet connection or Google Drive file ID.")

if __name__ == '__main__':
    main()
