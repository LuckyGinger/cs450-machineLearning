import time


def main():
    import os, webbrowser
    iexplore = os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"),
                            "Internet Explorer\\IEXPLORE.EXE")
    browser = webbrowser.get(iexplore)

    with open("../images/fall11_urls.txt", encoding="utf8") as infile:
        the_index = 0
        for index, line in enumerate(infile):
            the_index = index
            if the_index % 100000 == 0:
                print(the_index, line.split()[1])
            # url = line.split()[1]
            # print(index, url)
            # browser.open(url)
            # time.sleep(3)
            # os.system("TASKKILL /F /IM IEXPLORE.EXE")
        print(the_index)

if __name__ == '__main__':
    main()
