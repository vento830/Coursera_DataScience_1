####Add numbers###
x = 1
y = 2
print(x + y)

###in function###
def add_numbers(x, y, z):
    return x + y + z

print(add_numbers(1,5,2))

###standard value###
def add_numbers2(x,y, z=None):
    if (z==None):
        return x + y
    else:
        return x + y + z
print(add_numbers2(1,2,3))
print(add_numbers2(1, 4))

###Variabel###
a = add_numbers2
print(a(1,2))