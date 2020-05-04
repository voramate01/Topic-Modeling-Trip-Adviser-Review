
import scrapy

from Example.items import ReviewItem

hotel_list1_10 = ['https://www.tripadvisor.com/Hotel_Review-g60898-d89475-Reviews-Residence_Inn_Atlanta_Downtown-Atlanta_Georgia.html',
'https://www.tripadvisor.com/Hotel_Review-g60898-d1736251-Reviews-Courtyard_by_Marriott_Atlanta_Downtown-Atlanta_Georgia.html',
'https://www.tripadvisor.com/Hotel_Review-g60898-d673806-Reviews-The_Ellis_Hotel-Atlanta_Georgia.html',
'https://www.tripadvisor.com/Hotel_Review-g60898-d111374-Reviews-The_American_Hotel_Atlanta_Downtown_A_DoubleTree_by_Hilton-Atlanta_Georgia.html',
'https://www.tripadvisor.com/Hotel_Review-g60898-d565864-Reviews-Glenn_Hotel_Autograph_Collection-Atlanta_Georgia.html',
'https://www.tripadvisor.com/Hotel_Review-g60898-d86241-Reviews-Doubletree_Hotel_Atlanta_North_Druid_Hills-Atlanta_Georgia.html',
'https://www.tripadvisor.com/Hotel_Review-g60898-d86260-Reviews-The_Ritz_Carlton_Atlanta-Atlanta_Georgia.html',
'https://www.tripadvisor.com/Hotel_Review-g60898-d114387-Reviews-The_Westin_Peachtree_Plaza_Atlanta-Atlanta_Georgia.html',
'https://www.tripadvisor.com/Hotel_Review-g60898-d111335-Reviews-AC_Hotel_Atlanta_Downtown-Atlanta_Georgia.html',
'https://www.tripadvisor.com/Hotel_Review-g60898-d9600353-Reviews-Hotel_Indigo_Atlanta_Downtown-Atlanta_Georgia.html']


class GetReviewFollow(scrapy.Spider):
    name = 'hotel_1_10'
    allow_domains = ['www.tripadvisor.com']
    start_urls = hotel_list1_10
    custom_settings = { 'CLOSESPIDER_ITEMCOUNT': 8000 } #Get only 500 reviews
    
    def parse(self, response):
        Follow_reviews = response.css('.partial_entry::text').extract()
        
        for item in (Follow_reviews):
            Reviews = ReviewItem()
            Reviews['Review'] = item
            yield Reviews
            
            
       
        next_page = response.css('.ui_pagination a.ui_button::attr(href)').extract()[-2]
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)



#//*[@id="REVIEWS"]/div[8]/div/span[2]
            
            
