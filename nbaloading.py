
import io
import os
import boto3
import pandas as pd
import pyarrow.parquet as pq 
   
class DataLoader():
    dataframe_dictionnary= {'df_players':"BasketrefBoxscores.parquet/ceb09b1cf1f34cc482ead718f9d95147.snappy.parquet"
    , 'df_games':"BasketrefGames.parquet/61d3755ef1d348759c292b49676c3a76.snappy.parquet"}
    bucket = "betclic-data-test" 
    
    def load_pickle(self, file_name):
        return pd.read_pickle(file_name) 
            
    def load_data_with_parquet(self, file_name):  
        path_data = self.dataframe_dictionnary[file_name]
        bucket_uri = f"s3://{self.bucket}/{path_data}" 
        return pd.read_parquet(bucket_uri)
        
    def load_data_with_credentials(self, file_name):  
        credentials_file_name = 'credentials.txt' 
        credentials = open(credentials_file_name).read().split('/')
        session = boto3.Session(credentials[0], credentials[1])
        path = self.dataframe_dictionnary[file_name]
        buffer = io.BytesIO() 
        s3_object =  session.resource('s3').Bucket(self.bucket).Object(path)
        s3_object.download_fileobj(buffer)
        table = pq.read_table(buffer)
        df = table.to_pandas()
        return df 

    def load(self, file_name): 
        df = None
        try:
            print('  Info : Try to load a previously stored {} file'.format(file_name))
            df = pd.read_pickle(file_name)
            print("{} file loaded".format(file_name))
            return df
        except:
            print("  Info : No {} pickle File".format(file_name))
        
        if df is None:
            try:
                print('  Info : Try to download data with original method')
                df = self.load_data_with_parquet(file_name)
                df.to_pickle(file_name)
                return df  
            except:
                print('  Info : Can not load data without login. Please configure your Boto3 connexion.')
        
        if df is None:
            try:
                print('  Info : Try to download data with as a logged S3 user')
                df = self.load_data_with_credentials(file_name)
                df.to_pickle(file_name) 
                return df  
            except:
                print('  Info : Please set a credentials.txt file filled with {aws_access_key_id}/{aws_secret_access_key}')
              
