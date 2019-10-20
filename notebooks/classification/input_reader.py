from pyarrow.parquet import ParquetFile
import os


class InputReader:
    def __init__(self, input_parquets):
        self.input_parquets = input_parquets

    def read_parquet_files_as_row_groups(self, callback):
        parquet_partitions = []

        for input_parquet in self.input_parquets:
            parquet_partitions.extend(self.get_all_partitions_of_parquet(input_parquet))

        print("Total number of parquet partitions: %d" % len(parquet_partitions))

        for part_pq in parquet_partitions:
            self.process_single_parquet_partition(part_pq, callback)

    @staticmethod
    def get_all_partitions_of_parquet(input_parquet):
        parquets_list = []
        for file in os.listdir(input_parquet):
            if file.endswith(".parquet"):
                parquets_list.append(os.path.join(input_parquet, file))
        print("Parquet file: %s has %d partitions!" % (input_parquet, len(parquets_list)))
        return parquets_list

    @staticmethod
    def process_single_parquet_partition(parquet_location, callback):
        parquet_file = ParquetFile(source=parquet_location)
        num_row_groups = parquet_file.num_row_groups

        print("----------------------------------------------------------------------------------")
        print("%d row groups for partition: %s" % (num_row_groups, parquet_location))

        for index in range(0, num_row_groups):
            row_df = parquet_file.read_row_group(index, columns=["id", "img_binary"]).to_pandas()
            print(row_df.info(verbose=True))
            callback(row_df)


