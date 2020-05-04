# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html


class ExamplePipeline(object):
    def process_item(self, item, spider):
        return item


###########################################
import pymongo



class MongoPipeline(object):

    collection_name = 'REVIEW'

    def __init__(self, mongo_uri, mongo_db):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db

    @classmethod
    def from_crawler(cls, follow_getreview):
        return cls(
            mongo_uri=follow_getreview.settings.get('mongodb://localhost:27017/tripadvisor'),
            mongo_db=follow_getreview.settings.get('MONGO_DATABASE', 'items')
        )

    def open_spider(self, spiders):
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]

    def close_spider(self, spiders):
        self.client.close()

    def process_item(self, item, spider):
        self.db[self.collection_name].insert_one(dict(item))
        return item


#class MongoDBPipeline(object):

 #   def __init__(self):
  #      connection = pymongo.MongoClient(
   #         settings['MONGODB_SERVER'],
    #        settings['MONGODB_PORT']
     #   )
      #  db = connection[settings['MONGODB_DB']]
       # self.collection = db[settings['MONGODB_COLLECTION']]

 #   def process_item(self, item, spider):
  #      for data in item:
   #         if not data:
    #            raise DropItem("Missing data!")
     #   self.collection.update({'url': item['url']}, dict(item), upsert=True)
      #  log.msg("Reviews added to MongoDB database!",
       #         level=log.DEBUG, spider=spider)
        #return item