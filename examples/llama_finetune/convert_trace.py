import argparse

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Convert raw file to tool data")
    parser.add_argument("--input_file", help="Input file path")
    parser.add_argument("--output_file", help="Output JSON file path")

    # Parse the arguments
    args = parser.parse_args()

    # Process the input file
    print("\033[32mImport TensorFlow...\033[0m")
    import tensorboard_plugin_profile.convert.raw_to_tool_data as rttd
    input_file = args.input_file
    print("\033[32mXSpace to Tool Data...\033[0m")
    tv = rttd.xspace_to_tool_data([input_file], "trace_viewer^", {'tqx': ''})

    if isinstance(tv, tuple):
        tv = tv[0]

    # Write the processed data to the output file
    output_file = args.output_file
    print("\033[32mWriting file...\033[0m")
    with open(output_file, "w") as f:
        f.write(tv)

    print("\033[32mDone!\033[0m")

if __name__ == "__main__":
    main()