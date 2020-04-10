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

###Data formats###
#tuple
t=(1, 'a', 2, 'b')
l=[1,'a', 'b', 2]
print(type(t))
print(type(l))
l.append(3.3)
for item in t:
    print(item)
for item in l:
    print(item)

###Dictionarys and Format function###
#Dictionary
sales_record = {
    'price': 3.24,
    'num_items': 4,
    'person': 'Chris'}
sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'
print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['price'],
                             sales_record['num_items']*sales_record['price']))

###date and time###
import datetime as dt
import time as tm

print(tm.time())
dtnow = dt.datetime.fromtimestamp(tm.time())
print(dtnow)
print(dtnow.year, dtnow.month, dtnow.day)
print(dt.date.today())

people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

def split_title_and_name(person):
    return person.split()[0] + ' ' + person.split()[-1]

#option 1
for person in people:
    print(split_title_and_name(person) == (lambda x: x.split()[0] + ' ' + x.split()[-1])(person))


#option 2
list(map(split_title_and_name, people)) == list(map(lambda person: person.split()[0] + ' ' + person.split()[-1], people))

###list comprehesnion####
my_list = [number for number in range(0,1000) if number % 2 == 0]
print(my_list)
def times_tables():
    lst = []
    for i in range(10):
        for j in range (10):
            lst.append(i*j)
    return lst

print(times_tables() == [j*i for i in range(10) for j in range(10)])

####Numpy####
import numpy as np
y = np.array([4,5,6])
x = np.ones(3)
m = np.array([[7,8,9], [10,11,12]])
print(m)
#array shape (rows, columns)
print(m.shape)
#Count up in 2 stps
n = np.arange(0,30,2)
print(n)
#reshape array
n = n.reshape(3,5)
print(n)
o = np.linspace(0,4,9)
print(o)

print(x+y)
print(x.dot(y))
z= np.array([y, y**2])
print(z)
print(len(z))
print(z.T)

v = ['a', 'b', 'c']
w = [1,2,3]
print(v+w)

print(v[::2])
ty = np.arange(0,36,1)
print(ty)
print(ty[::2], ty[::3], ty[10:40:6])