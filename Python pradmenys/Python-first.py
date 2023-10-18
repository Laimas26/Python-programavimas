# print('Hello World\n' * 50)

x, y, z = "Test1", "Test2", "Test3"

print(x, y, z)

fruits = ["Apple", "Banana", "Orange"]

x, y, z = fruits

print(x, y, z)

# Global variable x
x = 5

def myFunc():
    # Galima ir funkcijoje nustatyti global variable su global komanda
    global y
    y = 111
    # global x
    # x = 222
    x = 21
    print("Number is", x)
    
myFunc()

print("Number is outside function is", x)

# Isspausdina kintamojo tipa
print(type(x))
