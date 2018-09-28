image_dir = '../pneumonia_data/result_images/'

files = dir([image_dir '*.jpeg']);
maxval = 30 % 256
threshold = 0.5
rmean = 0
nmean = 0
for file = files'
    im = imread([image_dir file.name]);
	im = imrotate(im,-90) % the image gets rotate shile we do the python stuff. revert is now
	rmean += mean(mean(im))
	nmean += 1
	im /= maxval;
	im = im > threshold;
	if max(max(im)) > 0
		imshow(im)
	end
end
rmean/nmean