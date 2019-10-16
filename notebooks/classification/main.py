# COMMAND ----------


from pyspark.sql import SparkSession
from pyspark.sql.types import BinaryType, StructType, StringType, StructField
from PIL import Image
from torchvision import transforms

import time
import io


# COMMAND ----------


INPUT_IMAGES_PARQUET = "/home/corneliu/downloaded_images.parquet"
CLASSIFIED_IMAGES_PARQUET = "/home/corneliu/classified_images.parquet"

MEAN = [0.485, 0.456, 0.406]
STANDARD_DEVIATION = [0.229, 0.224, 0.225]


# COMMAND ----------


spark = SparkSession.builder.master("local[*]").appName("Images downloader").getOrCreate()
sc = spark.sparkContext


# COMMAND ----------


def preprocess_image(row, transform_function):
    if row.image_bytes is None:
        return

    byte_array_image = bytearray(row.image_bytes)
    image = Image.open(io.BytesIO(byte_array_image))

    try:
        normalized_tensor = transform_function(image)
    except Exception as e:
        return




# COMMAND ----------


def read_downloaded_images(input_parquet):

    transform_function = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)]
    )

    input_parquet_schema = StructType([
        StructField("id", StringType()),
        StructField("image_bytes", BinaryType())
    ])

    parquetReadStream = spark\
        .readStream\
        .option("latestFirst", True)\
        .schema(input_parquet_schema)\
        .parquet(input_parquet)

    query = parquetReadStream.writeStream \
        .outputMode("append") \
        .format("console")\
        .start()

    query.awaitTermination()



read_downloaded_images(INPUT_IMAGES_PARQUET)