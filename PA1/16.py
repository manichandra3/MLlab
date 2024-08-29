import numpy as np
import pandas as pd
from PIL import Image


def image_to_matrix(image_path):
    img = Image.open(image_path)

    img_gray = img.convert('L')

    img_array = np.array(img_gray)

    return img_array


def save_matrix_to_csv(matrix, output_path):
    df = pd.DataFrame(matrix)

    df.to_csv(output_path, index=False, header=False)


def main():
    image_path = 'bird_small.png'

    output_path = 'output_matrix.csv'

    image_matrix = image_to_matrix(image_path)

    save_matrix_to_csv(image_matrix, output_path)

    print(f"Image matrix saved to {output_path}")


if __name__ == "__main__":
    main()
