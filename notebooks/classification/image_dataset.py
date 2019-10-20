from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import io


class ImageDataset(Dataset):

    MEAN = [0.485, 0.456, 0.406]
    STANDARD_DEVIATION = [0.229, 0.224, 0.225]

    image_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)
    ])

    def __init__(self, dataframe):
        self.tensors = ImageDataset.create_tensors_array(dataframe)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        return self.tensors[index][0], self.tensors[index][1]

    @staticmethod
    def convert_image_bytes_to_image_tensor(image_bytes):
        image_bytearray = bytearray(image_bytes)
        image = Image.open(io.BytesIO(image_bytearray))
        try:
            return ImageDataset.image_normalize(image)
        except:
            return None

    @staticmethod
    def create_tensors_array(dataframe):
        image_tensors = []
        for index, row in dataframe.iterrows():
            image_id = row['id']

            if row['img_binary'] is None:
                continue

            image_tensor = ImageDataset.convert_image_bytes_to_image_tensor(row['img_binary'])
            if image_tensor is None:
                continue

            image_tensors.append((image_id, image_tensor))

        return image_tensors