Notes of joy of computing in python
week 2 
install anaconda 
overview of print statement and interactive nature of python console
python inserts spaces between variables in print statement i.e print(a,b) will print a b 
Using ; to use sequence of instruction in console
using input to get users input
by default input in py is string even you give input a number
learned for , while and if

week 3 
list , operations : append , insert(at,item), .count(item),len(),.sort,.reverse ,arr[start:end+1]
in stats library we have trim_mean(arr,fraction)
 import matplot.pyplot  as plt
 plt.plot([x values],[y vlaues],"how to plot ")
 plt.yplot("name of y axis")
 
 code for ploting 
 import statistics
 import matplotlib.pyplot as plt
 Estimates =[1000,.....]
 y=[]
 Estimates.sort()
 tv=int(0.1*(len(Estimates))) // trim value
 Estimates =Estimates[tv:]
Estimates=Estimates[:len(Estimates)-tv]
for i in range(len(Estimates)):
    y.append(5)
plt.plot(Estimates,y,'r--')
plt.plot([statistics.mean(Estimates)],[5],'ro')
 plt.plot([statistics.median(Estimates)],[5],'bs')
 plt.plot([375],[5],'g^')
 
 random.choice(array), will return a random element from the array
 random.sample(word, len(word))
 This takes a sample of all the characters in word (i.e., it selects each character once) and returns them in a shuffled order.
 ''.join(...):
 The shuffled characters from random.sample are joined back into a string using ''.join(...). Here, '' is the separator, meaning the characters are joined without any space in between.
 
 File handling 
 with open("fileName.txt","r+") as myfile :
     print (myfile.read())
     myfile.write("write what you want")
myfile.close()

random library function 
randrange(1,n) , give 1 to n-1
randint(1, n) give 1 to n 
choice(arr), give a element from arr


Python passes mutable objects like lists by reference, meaning when you pass a list to a function, the function can modify the original list. However, technically speaking, Python follows a "pass-by-object-reference" or "pass-by-assignment" model, where the function receives a reference to the object but cannot rebind the original variable name.

In the code you shared, the list x is being passed to the evolve function, and any modifications to x[ind] inside the function directly affect the original list because lists are mutable.

In contrast, immutable objects (like integers, strings, and tuples) are passed in such a way that their value cannot be altered outside the function.

Points from assignment 
numbers_list = list(map(int, input_string.split()))
This line of code converts a space-separated string of numbers into a list of integers. Here's a breakdown:

input_string.split():

Splits the input_string by spaces (the default behavior of .split()).
For example, if input_string is "1 2 3 4", it splits it into a list of strings: ['1', '2', '3', '4'].
map(int, ...):

Applies the int function to each element of the list produced by .split(), converting each string to an integer.
So ['1', '2', '3', '4'] becomes [1, 2, 3, 4].
list(...):

Converts the map object (which is an iterator) into a list.

def remove_duplicates(numbers):
    seen = set()
    unique_numbers = []
    for number in numbers:
        if number not in seen:
            seen.add(number)
            unique_numbers.append(number)
    return unique_numbers

input_string = input("give the sequence")

numbers_list = list(map(int, input_string.split()))

unique_numbers = remove_duplicates(numbers_list)
print(" ".join(map(str, unique_numbers)), end="")

week 4
magic square : sum of any row , col and diagnoal of nXn matrix is equal, that sum is called magic number
magic numbers (n*(n^2+1))/2
creating 2d list :
    M-1 : 
        magicSquare=[]
        for i in range(n):
            l=[]
            for j in range(n):
                l.append(-1)
            magicSquare.append(l)
    M-2:
        magicSquare=[[-1 for i in range(n)] for j in range(n)]
        
algo for magic squar
postion of 1 is n/2 ,n-1
then of each next value , x=x-1, y=y+1
if x<0 then x= n-1,
if y==n then y=0,
if x<0 and y==n then x=0 , y=n-2,
if x,y is already occupied then y=y-2,x= x+1,

/ division, // integer division 

dobble game :
    used string.ascii_letters to get all the characters , dont forget to use list
    used random library 
BirthDay paradox:
    import datetime
import random

birthday = []
i = 0
while i < 50:
    year = random.randint(1895, 2017)

    # Check leap year
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        leap = 1
    else:
        leap = 0

    month = random.randint(1, 12)

    if month == 2 and leap == 1:
        day = random.randint(1, 29)  # Leap year, February has 29 days
    elif month == 2 and leap == 0:
        day = random.randint(1, 28)
    elif month in [4, 6, 9, 11]:
        day = random.randint(1, 30)  # These months have 30 days
    else:
        day = random.randint(1, 31)  # Other months have 31 days

    # Generate the correct day of the year
    dd = datetime.date(year, month, day)
    day_of_year = dd.timetuple().tm_yday

    # Append day_of_year to birthday list
    birthday.append(day_of_year)
    i += 1

# Sort the birthdays
birthday.sort()

print(birthday)
a brief overview of datetime library
Week 5
speech recogination 
import speech_recognition as sr
audio_file=("audiofilename.wav")
r=sr.Recognizer()
with sr.AudioFile(audio_file) as source :
    audio =r.record(source)
    
try :
    print("audio file has "+r.recognize_google(audio))
except sr.UnknownValueError:
    print("google speech recoginition could not understand audio")
execpt sr.RequestError:
    print("request error")

Learned about dictionaries in python
 various method like pop(), .values etes
 
    week 6 
    uses methods of strings like ascii_letters, substring ,and replace function, 
    learned recursion by fibo, factorial and binary search
    
    week 7
    learned image opening by PIL libray method called Image
    swapping a,b=b,a is valid
    spiral traversal
    turtle library for animation
    
Gps
reading form csv file and plotting location on gmap
imort csv
from gmplot import gmplot
gmap=gmplot.googleMapPlotter(28.689169,77324448,17)
gmap.coloricon ="http://www...."
with open ('route.csv') as f:
    reader =csv.reader(f)
    k=0
    for row in reader :
        lat=float(row[0])
        long =float(row[1])
        if(k==0):
            gmap.marker(lat,long,'yellow')
        else :
            gmap.marker(lat,long, 'blue')
gmap.marker(lat,long,'red')
gmap.draw("mymap.html")

week 8
tuples 
a=(48,439)
immutable
operations : len, count, index

lottery simulation
image processing:
    image flip:
        from PIL import Image
        img=Image.open('obatained.png')
        transposed_img=img.transpose(Image.FLIP_LEFT_RIGHT)
        transposed_img.save('corrected.png')
        print("done")
        png format is used for lossless compression
        pixelValue() gets the rgb of desired pixel

    enhance image
    import cv2
    img=cv2.imread('crime.png')
    clahe=cv2.createCLAHE()
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    enh_img=clahe.apply(gray_img)
    cv2.imwrite("enhanced.png",enh_img)
    
sorted(arr) give sorted version of arr
facebook sentiment analysis
import pandas library , provides easy to use data structure for data analysis
import nltk library , used to process human language,provides ananlysis of human data ,sentiment analysis involves working out wheather a piece of text is positive or negative or neutral
use vader library , gives intensity of sentiment 
 ord() function converts a character into its ascii notations
 
 import pandas as pd
 import nltk
 from nltk.downloader.download('vader_lexicon')
 file ='path'
 xl=pd.ExcelFile(file)
 dfs=xl.parse(xl.sheet_names[0])
 dfs=list(dfs['Timeline'])
 print(dfs)
 sid=SentimentIntensityAnalyzer()
 str1='UTC+05:30'
 for data in dfs :
     a=data.find(str1)
     if(a==-1):
         ss=sid.polarity_scores(data)
         print(data)
         for k in ss:
             print(k,ss[k])
             
graphs in python
import networkx as nx
import matplotlib.pyplot as plt
G=nx.Graph()
G.add_node(1) // to add node one by one 
G.add_nodes_from(arr)
G.add_edge(2,3)
print(G.nodes())
print(G.edges())
nx.draw(G)
plt.show()

G=nx.gnp_random_graph(20,0.2) random graph
G=nx.complete_graph(4) complete graph of 4 vertices

learned about barabasi_albert_graph()
learned about gexf format for giphi

reading edge list :
    import networkx as nx
    import numpy
    G =nx.read_edgelist("file.txt")
    N=list(G.nodes())
    spll=[]
    for u in N:
        for v in N:
            if(u!=v):
                l=nx.shortest_path_length(G,u,v)
                spll.append(l)
    
    min_spl=min(spll)
    max_spl=max(spll)
    avg_spl=numpy.average(spll)
    
import numpy as np
from PIL import Image
width =5
height=4
array=n.zeros([height,width,3],dtype=np.unint8)
img=Image.fromarray(array)
img.save('test.png')
array1-np.zeros([100,200,3],dtype=np.unint8)
array1[:,:100]=[255,128,3]
array1[:,100:]=[0,0,255]
img=Image.fromarray(array1)
img.save("image.png")

program to get rgb value of a desired pixel
from PIL import Image
im=Image.open("test1.png")
rgb_im=im.convert('RGB')
r,g,b=rgb_im.getpixel((150,1))
print (r,g,b)

calculating area without measure
import scipy.misc 
from PIL import Image 
import numpy as np
import random 
img=scipy.misc.imread("file.png")
count_pun =0 # calculating sample count of punjab
count_ind=0  # calculating sample count of india
count=0
while(count <=10000):
    x=random.randint(0,2735)
    y=random.randint(0,2480)
    z=0 
    if(img[x][y][z]==60):
        count_ind+=1 
        count+=1 
    elif(img[x][y][z]==80):
        count_pun+=1
        count+=1 
        
area_punjab=(count_pun/count_in)*328763

week 10 
josephus problem

numpy is intended to handle matrices 

numpy.zeros((r,c)) give matrix of rxc with zeros
arr created by numpy.array , arr.shape give dimensions 

  we can provide dtype=int64 also 
operation we can perform on matrix created by numpy:
    x+y
    x/y
    numpy.multiply(x,y)
    numpy.sqrt(x)
    numpy.sum(x)
    
Data compression 
making image to matrix
import numpy 
from PIL import Image
im=Image.open('image.jpg')
pixelMap=im.load()
I=numpy.asanyarray(Image.open('image.jpg'))
img=Image.new(im.mode,size)  # new copy of above image
pixelNew=img.load()
now we can perform many techniques like modulo or maping on each pixel  and  compress the image (loosy compression)

Assignment points
Week 2 :
    print() statement by default prints in next line, end="" can be also used as side parameter
    for loops in python dont have conditions :
    python dont have do while loop by default , you have to implenment it using variable
    abs() exist in py
Week 3 :
    list can store different data types
    with open("file.txt",mode) :
        something something
        
    will take care of automatic closing of file
    The map() function is used to apply a given function to every item of an iterable, such as a list or tuple, and returns a map object (which is an iterator),The map() function returned an iterator, which we then converted into a list using list(). This is a common practice when working with map()
    The split() method splits a string into a list.
You can specify the separator, default separator is any whitespace.
   seen =set(), not in seen 
   subarray/slice [:],reversed list  [::-1], 
   arr.index(value) return index
   
   
week4 :
    Magic constant : n(n**2+1)/2
    Transposed matrix of magic square is magic square
    peigon hole principle
    Ramanujan magic square :139
    week 5: 
    Monty hall problem , probab after swapping the choice is 0.66
    .wav file contain audio file
    
Week 5:
    arr.extend(arr2), adds all the elements of arr2 to the last of arr
    indexing in python is .... -2, -1 
    fucntion overloading is not supported in python , latest defination is taken
    Could we check frequency of letters in a long ciphertext and map them to frequency of letters in English to decrypt the message?

Hint: Search the internet for more info, if needed.
 Yes, it is possible.
 Dictonary methods :
     from pythons perspective , dictionaries are defined as objects with the data type 'dict'
     using dict constructor for initialisation :
         this_dict=dict(name:"jhon", age=36,country ="Norway")
         dict.items(), dict.keys(),dict.update(dict2),dict.values(),dict.has_key(key)
         ceaser Cipher : The Caesar cipher is a classic example of ancient cryptography that involves shifting each letter of a plaintext message by a certain number of letters, historically three. It is a type of substitution cipher where one letter is substituted for another in a consistent fashion.
week 7:
    csv file: coma seprated values, to read it use csv module
    Working with csv in python :
        Reading:
            import csv
            fileName="file.csv"
            fields=[]
            rows=[]
            with open(fileName,'r') as csvfile:
                #creating a csv reader object
                csvreader=csv.reader(csvfile)
                fields=next(csvreader) # returns a list and advances to next lines
                for row in csvreader:
                    rows.appendd(row)
            Reading csv into a dictionary 
            import csv 
            with open(filename,'r') as csvfile:
                csvreader=csv.DictReader(csvfile)
                data_list=[]
                for row in csv_reader:
                    data_list.append(row)
    Installing package using pip:
        pip install package_name
        gmplot libarary :
            to plot data on google maps
            Basic function of gmplot:
                create base map using lat and long coordinate:
                    gmap1=gmplot.GoogleMapPlotter(30.31,78.443947894,13)
                    gmap1.draw("path")
                    or 
                    gmap2=gmplot.GoogleMapPlotter.from_geocode("Dehrdun, India")
                    gmap2.draw("path")
            Turtle library :
                used to draw 
                    import turtle
                    turtle=turtle.Turtle()
                    turtle.forward(amount) moves forward 
                    turtle.backward(amount)
                    turtle.right(degree) #clockwise
                    turtle.left(degree) # anticlockwise 
                    turtle.penup() #picksup the turtle's Pen,i.e not draw
                    turtle.pendown() #put down the turtle's pen i.e draw
                    turtle.color() #changes the pens color
                    turtle.dot() #leaves dot at the current position
                    
                    drawing circle 
                    import turtle
                    screen =turtle.Screen()
                    screen.bgcolor("white")
                    pen = turtle.Turtle()
                    pen.speed(0)
                    pen.fillcolor("red")
                    pen.begin_fill()
                    pen.circle(100)
                    pen.end_fill()
                    pen.hideturtle()
                    turtle.done()  
                PIL library :
                    Imaging library 
                    from PIL import Image
                    img=Image.open("image.png")
                    img.show()
                    print(img.mode) #give the mode of image pixels i.e 1(1 bit pixel) L(8 bit pixels ) P(8 bit pixels mapped to any other mode ) RGB(3X8 bit pixel) RGBA(4X8 pixles)
                    print(img.size) # here it have width and height
                    print(img.format)
                    rotated_img=img.rotate(angle)
                    
                    new_img=img.resize(size) # size=(39,49) w h 
                    img.save("DesiredName.png")
                    
                    
                    to check an element is present in list or not 
                    if a in arr :
                        yes
                    elif b in arr2:
                        yes
                    else : 
                        no
        Week 8 :
            Different Operations Related to Tuples
            Below are the different operations related to tuples in Python:

            Concatenation t=t1+t2
            Nesting  t=(t1,t2)
            Repetition t=('s',)*3
            Slicing t[1:]
            Deleting del t
            Finding the length len(t)
            Multiple Data Types with tuples t=('t',1,True)
            Conversion of lists to tuples t=tuple(list)
            Tuples in a Loop for i in range(n):
                t=(t,)
                
            Matplotlib :
                used to plot lines , graph etc
                creating line plots:
                    import matplotlib.pyplot as plt
                    x=[]
                    y=[]
                    plt.plot(x,y)
                    plt.title("line chart")
                    plt.xlabel("x-axis")
                    plt.ylable("y-axis")
                    plt.show()
                    Why Two Lines Appear
With x = [3, 1, 3] and y = [3, 2, 1], the plot goes:

From (3, 3) to (1, 2).
Then from (1, 2) to (3, 1).
Since the x-values jump from 3 to 1 and back to 3, matplotlib creates a zigzag pattern instead of a single continuous line in one direction.

stem plot in matplotlib 
plt.stem(x,y,use_line_collection=true)
plt.show()
 for bar graph , use bar():
for histogram plt.hist(x,bins=[put range vlaues]) :
for scatter(x,y) # will plot scatter plot
for pie chart :import matplotlib.pyplot as plt 
# data to display on plots 
x = [1, 2, 3, 4] 
e  =(0.1, 0, 0, 0)
plt.pie(x, explode = e)
plt.title("Pie chart")
plt.show()

to get ascii value ord('a')
vader is used for sentiment analysis
    a, b = map(int, input().split())              
      list comprehension ([char.lower() for char in s if char.isalpha()])      
      
week 9 
nltk library :
   it provides tools for working with human language data .
Tokenization:
    nltk.word_tokenize(text)
    nltk.sentence_tokenize(text)
    
    stemming , lemmatization ,sentimentAnalysis
    import nltk 
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    text =""
    lemmatizer=WordNetLemmatizer()
    stemmer=PorterStemmer()
    for word in text:
        print(word,stemmer.stem(word),lemmatizer.lemmatize(word,'v'))
        
    sia=SentimentIntensityAnalyzer()
    print(sia.polarity_scores("hello how are you"))
     corpus module in nltk :  It allows users to work with large bodies of text in a structured way, facilitating tasks like natural language processing, machine learning, and linguistic analysis            
  nltk.corpus.stopwords() removes stopwords from string
  nltk.wordnet is used for synonyms and antonyms
  nltk.chunk is used for Named Entity Recoginition
   ntlk.pos_tag is used for part of speech tagging
   
  
    
  
    
  
    NetworkX library :
        NetworkX is package for creation , manipulation, and study of the complex network 
        adding a node and edges :
            G.add_node(1)
            G.add_nodes_from([])
            G.add_edge(1,2)
            G.add_eges_from([(1,2),(2,3)])
            G.remove_node(1)
            G.remove_edge(1,2)
            print(G.edges)
            print(G.nodes())
            print(G.degree(2))
            print(G.neighbors(2))
            nx.info(G) , get a summary of the graphs structure
            nx.shortest_path(G,s,e)
            nx.connected_components(G), get all the connected compo in undirected graph
            nx.draw(G)
            nx.read_adjlist("filepath"), nx.write_adjlist(G,'filepath')
            nx.read_edgelist("file_path"),nx.write_edgelist(G,'filepath')
            nx.spring_layout(G)
            nx.clustering(G)
            nx.betweenness_centrality(G)
            
    Gephi:
        Gephi is an open-source graph visualization and exploration tool widely used for analyzing and visualizing complex network structures.
        Area proportional nodes are a visualization technique where the size of each node in a network graph is adjusted according to a specific attribute or metric
        Key Metrics in Network Visualization: Node Degree, Weight, and Centrality
        Here's a summary of the steps to create area-proportional nodes in Gephi:

Step 1: Loading the Graph Data
Open Gephi and create a new project.
Import your graph data (e.g., CSV, GML) using "File > Open."
Follow the import prompts to correctly map nodes and edges.
Step 2: Calculating Node Metrics
In the "Statistics" panel, calculate node metrics like Degree, Betweenness Centrality, or Closeness Centrality by running each metric. These metrics will later be used to resize nodes.
Step 3: Applying Node Ranking
In the "Appearance" tab, go to the "Ranking" sub-tab.
Select a metric to rank the nodes (e.g., Degree).
Adjust node size parameters to set the minimum and maximum sizes, then apply the ranking.
Step 4: Adjusting Visualization
Use the "Layout" panel to arrange nodes with algorithms like ForceAtlas 2 or Yifan Hu.
Customize node and edge appearance (color, size, labels) in the "Appearance" panel.
Adjust edge properties and add labels for clarity.
Step 5: Exporting the Visualization
Export your graph in formats like PDF or PNG.
Save the project for future edits.
        
    stylometry :
        study of writing style, by analyzing the aspects like word choice, sentences length, punctuation frequency and grammatical patterns to determine.
        nltk: word_tokenize, FreqDist 
        
shorthand :
    total_length=sum(len(word) for word in words)
    total number of edges in complete graph (n*(n-1))/2 
    
Week 10 :
    string methods :
        strip() , remove leading and trailing whitespaces
        str.title()
        str.split()
        str.join(iterable)
        str.replace(old,new)
        str.find(substring)
        str.count(substring)
        str.startwith(prefix)
        str.capitalize()
        str.isalpha()
        str.isdigit()
        str.isalnum()
        str.capwords() , capitlize first letter of each word
        random.choice(str) , give random anagram of str
        string.Template() , used for formatting strings with placeholders in the string
Numpy:
np.array
arr[1,2]
numpy is used in image processing
Basic functions :
    Array creation :
        np.array () , create array from list or tuple
        np.zeros((m,n)) , creates an array with zeros 
        np.arange(10) ,creates an array with range of values
    Array Operations :
        np.add(a,b)
        np.subtract(a,b)
        np.multiply(a,b)
        np.divide(a,b)
        np.dot(a,b) # dot product
        np.linalg.inv(a) #inverse of matrix
        np.transpose(matrix)
        np.flatten()
        np.linalg.eig()
        np.where(condition eg arr>2)
        
    Array manipulation:
        arr=np.arnage(12).reshape((3,4))
        arr1=np.array([[],[]])
        arr=np.concatenate((a,b),axis)
        
        statistical functions :
            np.mean(arr)
            np.median()
            np.std()
            np.sum()
        slicing :
            arr[r1:r2,c1:c2]
            gives a subgrid from (r1,c1) to(r2-1,c2-1)
Image compression 
image compression is done by pil:
    img.save('filename','format',quality=60)
by opencv:
    img=cv2.imread('inputimge')
    cv2.imwrite('',img,[int(cv2.IMWRITE_JPEG_QUALITY),85])
by scikit-image:
    img=io.imread('file')
    img=img_as_ubyte(img)
    io.imsave('filename',img,quality=85)
    
    Some imp fucntions of PIL:
        Image.resize()
        Image.crop((l,u,r,l)) # left upper right lower
        Image.rotate(angle)
        img.transpose(Image.FLIP_LEFT_RIGHT)
        img.filter(ImageFilter.BLUR)
        ImageEnhance.Brightness(img)
        img.convert('L') # grey scale conversion
        img.info[]
        Image.getpixel((x,y))
        Image.putpixel((x,y),(r,g,b))
        
        
Week 11 :
    selenium:
        from selenium import webdriver
        browser=webdriver.Chrome("driverpath")
        to browe a link :
            browser.get("https://....")
        to get a element by name:
            elem=browser.find_element_by_link_text("Download")
         search=browser.find_element_by_id('q')
         
         whatsApp automation learnings :
            driver=webdriver.Chrome('')
            driver.get('https://')
            wait=WebDriverWait(driver,600)
            
            find element:
                driver.find_element(By.ID,"search") 
                driver.find_element(By.NAME,'submit')
                logo =driver.find_element(By.XPATH,"")
                driver.find_elements(By.CLASS_NAME,"item")
            ActionChain:
                action=ActionChains(driver)
                element=driver.find_element(By.ID,"hover-element")
                action.move_to_element(element).click().perform()
                action.send_keys(Keys.PAGE_DOWN).perform()
                
            WebdriverWait: wait unitl an element is visible or clickable
                wait=WebDriverWait(driver,10)
                element=wait.until(EC.visibilty_of_element_located((By.ID,"target")))
                element.click()
                
                Keys is a class in Selenium’s selenium.webdriver.common.keys module that provides constants for keyboard keys. These constants allow you to simulate keyboard actions, such as pressing the Enter key, Space, Tab, or any other key on the keyboard, within a web application.
   
        datetime library :
            datetime.date.today() # today date YYYY-MM-DD
            datetime.datetime.now() # YYYY-MM-DD HH:MM:SS
            datetime.datetime.now()+datetime.timedelta(days=7)
            datetime.strptime(date_str,"format like %Y-%m-%d") , takes date time string and gives date time object
            
            
        Calender library 
        calender.month(2024,1) , give january 2024 calender
        calender.isleap(year)
        calender.monthcalender(year,month) gives month calender
        calender.weekday(yyyy,mm,dd) gives weekday of that date
        calender.prcal(year)
        calendar.monthrange(year, month) gives no. of days in month of a year
        
        
        Collection module :
            The collections module includes:

Counter: Counts occurrences of elements.
defaultdict: Like a dictionary but with a default value if a key doesn’t exist.
OrderedDict: Maintains the order of items based on their insertion.
namedtuple: Creates tuple subclasses with named fields.
deque: A double-ended queue with fast appends and pops from both ends.
            
       counting=Counter(arr) 
       counting.keys() ,gives keys 
       counting.values , give vlaues
       counting.most_common(1),give highest frequency element
       coutning.items() givelist of element , count pair
       counting.update(arr) merges the 2 list
       
    The Collatz Conjecture, also known as the 3n + 1 problem, is a famous unsolved problem in mathematics that involves a sequence defined as follows:

Start with any positive integer 
𝑛
n.
If 
𝑛
n is even, divide it by 2.
If 
𝑛
n is odd, multiply it by 3 and add 1.
Repeat the process indefinitely.
The conjecture states that no matter what positive integer you start with, you will eventually reach the number 1.

PageRank is an algorithm developed by Larry Page and Sergey Brin, used primarily by Google to rank web pages based on their importance and relevance within the hyperlink structure of the web.

How It Works:

The web is represented as a directed graph, where web pages are nodes and hyperlinks are directed edges.
PageRank simulates a random surfer who navigates the web by following links. The algorithm calculates the probability of the surfer landing on a specific page.
The formula for PageRank incorporates a damping factor (typically 0.85) to account for the likelihood of the surfer randomly jumping to any page instead of following links.
Pages with more inbound links from important pages receive higher PageRank scores, while the quality of the linking pages also affects the ranking.
Applications:

PageRank is primarily used in search engines to deliver relevant search results.
Variations of the algorithm are applied in social network analysis and recommendation systems to assess influence and rank items based on user interactions.
Limitations:

The dynamic nature of the web requires regular updates to PageRank calculations.
PageRank can be manipulated through tactics like link farms, affecting its effectiveness.
Modern search engines now incorporate additional factors beyond PageRank to enhance the relevance of search results.

pytz library is used to deal with time zones
pytz.localize() is used to set time zone for a datetime object

Degree of separation is equivalent to The average length of the shortest path in a graph
