# This file

import json
import matplotlib.pyplot as plt
import numpy as np

# Review text list for each category
''' 
1 331786
2 224334
3 321700
4 674636
5 1132610
TOTAL review_cnt: 2685066 MAX_LENGTH: 5000 MAX_NUMBER: 2304
'''

N_stars = 5
star_length_to_counts = {}
star_absolute_freq = [331786, 224334, 321700, 674636, 1132610]
review_cnt = 2685066
max_length = 5000
max_number = 2304
colors = np.array(['#FF6666', '#FFB266', '#FFFF66', '#B2FF66', '#66FF66'])

def data_wrangling():
	global star_length_to_counts
	global star_absolute_freq
	global review_cnt, max_length, max_number


	star_reviews_dict = {} # Tuple (star, list of review text) e.g. (4, ['good service', 'Nice place!'])

	for i in range(N_stars):
		star_reviews_dict[i] = []
	
	with open('data/yelp_academic_dataset_review.json') as data:
		for line in data:
			review = json.loads(line)
			star = review['stars']
			text = review['text']
			star_reviews_dict[star-1].append(text)
			max_length = max(max_length, len(text))
			review_cnt += 1
			if (review_cnt % 100000 == 0):
				print 'review_cnt:', review_cnt

	for i in range(N_stars):
		star_absolute_freq[i] = len(star_reviews_dict[i])
		print i+1, star_absolute_freq[i]
		# Get text length and number of reviews of this length.
		star_length_to_counts[i] = [0] * (max_length+1)
		for review_text in star_reviews_dict[i]:
			star_length_to_counts[i][len(review_text)] += 1
			max_number = max(max_number, star_length_to_counts[i][len(review_text)])

	star_reviews_dict.clear()
	print 'TOTAL review_cnt:', review_cnt, 'MAX_LENGTH:', max_length, 'MAX_NUMBER:', max_number
	for i in range(N_stars):
		print 'star_length_to_counts', i, star_length_to_counts[i]

def freq_bar_chart():
	fig = plt.figure(figsize=(12,8))
	
	bar_indices = np.arange(N_stars)
	x_labels = np.array([x_stars+1 for x_stars in range(N_stars)])
	star_relative_freq = np.array(star_absolute_freq)/float(review_cnt)

	bars = plt.bar(bar_indices, star_relative_freq, width=1, color=colors)
	for (idx, rect) in enumerate(bars):
	    plt.gca().text(rect.get_x()+rect.get_width()/2., 1.05*rect.get_height(), '%d'%int(star_absolute_freq[idx]), ha='center', va='bottom')

	plt.xticks(bar_indices+.5, x_labels)
	plt.xlabel('Star Category')
	plt.ylabel('Relative Frequency')
	plt.ylim([0,1])
	plt.title('Star Category Distribution for {0} Reviews'.format(review_cnt))
	plt.grid(True)

	plt.show()

def length_number_chart():
	lists = []
	with open('data/stat.txt') as data:
		for line in data:
			tokens = line.split(', ')
			lists.append(tokens)
	
	plt.figure(figsize=(12,8))
	x = np.arange(0, max_length+1, 1);

	# plt.gca().set_color_cycle(colors)
	# for i in range(N_stars):
	# 	plt.fill(x, lists[N_stars - i - 1], color=colors[N_stars - i - 1], alpha=.9)
		# ax.fill_between(xs, ys, where=ys>=0, interpolate=True, color=colors[i], alpha='0.7')
	plt.fill(x, lists[4], color=colors[4], alpha=.7)
	plt.fill(x, lists[3], color=colors[3], alpha=.7)
	plt.fill(x, lists[0], color=colors[0], alpha=.7)
	plt.fill(x, lists[2], color=colors[2], alpha=.7)
	plt.fill(x, lists[1], color=colors[1], alpha=.7)

	plt.legend(['5 Star', '4 Star', '1 Star', '3 Star', '2 Star'], loc='upper right')

	# plt.fill_between(0, 0, line)
	# xtick_labels = 10 * np.arange(0, max_length, 10)
	plt.axis([0, max_length, 0, max_number])
	plt.grid(True)
	# ytick_labels = 10 * np.arange(0, max_number, 10000)

	
	# plt.xticks(x, xtick_labels);
	# plt.yticks(y, ytick_labels);
	plt.xlabel('Review Length')
	plt.ylabel('Number of Reviews')
	plt.title('Review Length vs. Number of Reviews')

	plt.show()


# Main funciotn.
def main():
	print 'Start running...'
	# data_wrangling()
	print 'Statistics finished...'
	# freq_bar_chart()
	length_number_chart()


# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()