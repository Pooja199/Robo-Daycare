import sys
import matplotlib.pyplot as plt

files = sys.argv[1:]

def is_number(n):
    try:
        float(n)  
    except ValueError:
        return False
    return True
# epoch = 1
vals = []
episode = []
last_episode = -1
for file in files:
	f = open(file, "r")
	lines = f.readlines()
	for line in lines:
		# if last_episode > 6000: 
			# break
		words = line.split()
		numbers = []
		for word in words:
			if(word.isnumeric()):
				numbers.append(word)
		if(int(numbers[0]) > 25000):
			break
		episode.append(int(numbers[0]))
		vals.append(int(numbers[2]))
		# ep = int(words[0])
		# if is_number(ep):# and epoch%5 == 0:
			# if ep > last_episode:
				# episode.append(ep)
				# last_episode = ep
			# else:
				# continue
		# if is_number(words[-1]):# and epoch%5 == 0:
			# vals.append(int(words[-1]))
			# if(int(words[0]) > 5000):
				# break
			# print("YES")
		# epoch += 1

# plt.plot(vals)
print(len(episode), len(vals))
# plt.ylim(top=900)
plt.plot(episode,vals)
plt.xlabel("Episodes")
plt.ylabel("Average Rewards")
plt.show()