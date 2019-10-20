import pyarrow as pa
import pandas as pd
import time

# COMMAND ----------


class OutputWriter:
    def __init__(self, output_parquet):
        self.output_parquet = output_parquet

    def write_to_parquet(self, classification_result):
        start_time_writing = time.time()
        classification_df = OutputWriter.convert_classification_to_dataframe(classification_result)
        table = pa.Table.from_pandas(classification_df)



        duration_writing = time.time() - start_time_writing

        print("Wrote the classification result to the parquet file: %s. Duration: %.2f" %
              (self.output_parquet, duration_writing))

    @staticmethod
    def convert_classification_to_dataframe(classification_result):
        print("Converting the classification result to pandas dataframe")

        df = pd.DataFrame(columns=['image_id', 'label', 'confidence'])

        for image_id, matched_classes in classification_result:
            for label, confidence in matched_classes.items():
                df.append({'image_id': image_id, 'label': label, 'confidence': confidence})

        print("Finished converting to pandas dataframe")

        return df
