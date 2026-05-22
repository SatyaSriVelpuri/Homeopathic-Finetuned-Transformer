import scrapy
import os
import json

class Sangama(scrapy.Spider):
    name ="Sangama"
    def getUrls():
        with open("sources.txt","r",encoding="utf-8") as f:
            return [line.strip() for line in f]
    start_urls = getUrls()

    def __init__(self, name = None, **kwargs):
        super().__init__(name, **kwargs)
        self.All_medicines = []

    def parse(self,response):
        for a in response.css("div.remedy_list a"):
            if 'anchor' not in a.attrib.get("class"," "):
                medicine = a.xpath("text()").get().strip().lower()
                if not medicine.startswith('-preface'):
                    med_url = response.urljoin(medicine.replace(" ", "-"))
                    yield scrapy.Request(
                        url=med_url, 
                        callback=self.parse_medicines, 
                        meta={"medicine": medicine}  
                    )
    def parse_medicines(self,response):
        medicine = response.meta.get("medicine")

        header_text = response.xpath("//h1/following-sibling::text()[normalize-space()]").getall()
        p_text = response.xpath("//p[not(ancestor::div[@id='remediaLink'])]//text()").getall()
        content = " ".join(header_text + p_text).strip()


        for medicine_data in self.All_medicines:
            if medicine_data["name"] == medicine:
                medicine_data["content"] += "\n" + content
                self.log(f"Updated {medicine} in memory")
                return
        
        self.All_medicines.append({"name": medicine, "content": content})
        self.log(f"Updated {medicine} in memory")




    def closed(self, reason):
        try:
            os.makedirs("medicines_data", exist_ok=True)  # Create a folder for storing the text files

            for medicine in self.All_medicines:
                filename = f"medicines_data/{medicine['name'].replace(' ', '_')}.txt"  
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(medicine["content"])  # Save only the content

            print("Data successfully saved as individual text files in 'medicines_data/'")
        except Exception as e:
            print(f"Error saving data: {e}")



if __name__ == '__main__':
    from scrapy.cmdline import execute
    execute(['scrapy', 'runspider', 'Sangama.py'])