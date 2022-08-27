
#Dette er bare for å lage en liste av 100 tilfelfeldige tall. Du trenger ikke å forstå
import random
lst = [random.randint(0, 100) for i in range(100)]

# 0 i python betyr False, 1 Betyr True 

# % deler på tallet som et heltall og gir rester

# f.eks 5%2 = 1
#  5/2=2 + 1 i rest

# 2%2 = 0
# 2 / 2 = 1, derfor 0 i rester

# Vi vet da at hvis det er 0 i rest, så er det partall, 
# men hvis det er 1 i rest, så er det oddetall. 

# Funksjonen under returnerer True eller False. 1 er True, 0 er False
# Oddetall får 1 i rest, så da returnerer den True for Oddetall

def odd(n):
  return n%2

# Not gir det motsatte, så hvis n%2 = 1 (True), så gir den False.
# Basically det motsatte av oddetall. 

def even(n):
  return not n%2

# Her så går vi bare gjennom listen av tilfeldige tall, 
# og printer de som er odd

"""
for i in lst:
  if odd(i): 
    print(i)
"""

for i in range(1, 101, 2):
  print(i)


def fun(a, b, c):
  return a * b * c

fun(1, 2, 3)