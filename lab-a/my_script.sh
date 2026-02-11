#!/usr/bin/env sh

# Display a welcome message
name="Alex"
echo "Hello $name!"

# Display a different message depending on the value of $hour
hour="$(date +"%H")"
if [ "$hour" -ge 12 ]; then
    echo "It is the afternoon."
else 
    echo "It is the morning."
fi

# Display colours of the rainbow
set -- "Red" "Orange" "Yellow" "Green" "Blue" "Indigo" "Violet"
for a in "$@"; do 
  echo $a
done

# Display the contents of /
for f in /*; do
  echo "/ contains $f"
done