# Display a welcome message
$name = "Alex"
Write-Host "Hello $name!"

# Display a different message depending on the value of $hour
$hour = Get-Date -Format HH
if ($hour -ge 12) {
    Write-Host "It is the afternoon."
} else {
    Write-Host "It is the morning."
}

# Display colours of the rainbow
$colors = @("Red","Orange","Yellow","Green","Blue","Indigo","Violet")
For ($i=0; $i -lt $colors.Length; $i++) {
    Write-Host $colors[$i]
}

# Display the contents of /
Get-ChildItem -Path / | ForEach-Object {
    Write-Host "/ contains: $_"
}