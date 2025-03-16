#!/bin/bash
#/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Database/run_scraper.sh
# Define paths
PROJECTS_DIR="/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Projects"
BLENDER_PATH="/Volumes/ProgramFiles/Apps/Blender_36.app/Contents/MacOS/Blender"
SCRIPT_PATH="/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Database/scraper_final_automated.py"

# Loop through each project folder
for PROJECT in "$PROJECTS_DIR"/*; do
    if [ -d "$PROJECT" ]; then  # Check if it's a directory
        echo "Processing project: $PROJECT"
        
        # Run Blender with the script and wait for it to finish
        "$BLENDER_PATH" --background --python "$SCRIPT_PATH" -- "$PROJECT"
        
        echo "Finished processing: $PROJECT"
        echo "--------------------------------"
    fi
done

echo "All projects processed!"