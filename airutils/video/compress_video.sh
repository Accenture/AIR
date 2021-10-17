ffmpeg -i $1 -vf "scale=iw/2:ih/2" -acodec copy -b:v 3000000 "${1%.*}_compressed.mov"
#ffmpeg -i $1 -acodec copy -b:v 8000000 "${1%.*}_compressed.mov"
