import pandas as pd
import numpy as np

p = print
animals = ['Tiger', 'Bear', 'Moose']
print(pd.Series(animals))

animals = ['Tiger', 'Bear', None]
print(pd.Series(animals))

numbers = [1, 2, None]
print(pd.Series(numbers))
print(np.nan == None)
p(np.nan == np.nan)
p(np.isnan(np.nan))

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
p(s)
p(s.iloc[3])
p(s.loc['Golf'])

original_sports = pd.Series({'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'],
                                     index=['Cricket',
                                            'Cricket',
                                            'Cricket',
                                            'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)
p(original_sports)
p('Cricket loving Countries:' + cricket_loving_countries)
p('All countries' + all_countries)
p('Loc' + all_countries.loc['Cricket'])

#### Data frame ####
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1,purchase_2,purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
p(df.head())
df = df.set_index([df.index, 'Name'])
df.index.names = ['Location', 'Name']
print(df)
df = df.append(pd.Series(data={'Cost': 3.00, 'Item Purchased': 'Kitty Food'}, name=('Store 2', 'Kevyn')))
p(df)


