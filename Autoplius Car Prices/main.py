import requests
from bs4 import BeautifulSoup


baseUrl = 'https://autoplius.lt/importhandler?datacollector=1&category_id=2&'


def handleInputs():
    make_id = input("Iveskite markes id:")
    
    return make_id



def main():
    connectionCheck()
    # make_id = handleInputs()
    # print(baseUrl + f"make_id={make_id}")
    # make_req_string = f"make_id={make_id}"
    make_req_string = ""
    
    getRequest(make_req_string)





def getRequest(make_req_string):
    # response = requests.get(baseUrl + make_req_string)
    response = requests.get("https://autoplius.lt/importhandler?datacollector=1&category_id=2&make_id=67&model_id=10988&external_id=24315650")

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        print(soup)
        # Now, 'soup' contains the parsed HTML of the webpage.
    else:
        print('Failed to retrieve the webpage. Status code:', response.status_code)
   




  
def connectionCheck():
    response = requests.get(baseUrl)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        print("Success")
        # Now, 'soup' contains the parsed HTML of the webpage.
    else:
        print('Failed to retrieve the webpage. Status code:', response.status_code)


main()



