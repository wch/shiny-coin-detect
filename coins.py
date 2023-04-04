#!/usr/bin/env python3

import cv2
import numpy as np


def detect_coins(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Find circles in the image
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=100,
        param2=80,
        minRadius=50,
        maxRadius=240,
    )

    # If no circles are detected, return an empty list
    if circles is None:
        return []

    # Round the circle parameters
    circles = np.round(circles[0, :]).astype("int")

    # Define a list to store results
    results = []

    max_r = max([r for _, _, r in circles])

    # Iterate over the detected circles
    for x, y, r in circles:
        r_ratio = r / max_r

        # Get the average color within the coin's region
        mask = np.zeros_like(image)
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        masked_image = cv2.bitwise_and(image, mask)
        avg_color = cv2.mean(masked_image)

        # Adjust the coin type estimation based on the new radius range and color
        if 0.9 <= r_ratio and r_ratio < 1.1:
            coin_type = "Quarter"
        elif 0.8 <= r_ratio and r_ratio < 0.86:
            coin_type = "Nickel"
        elif 0.68 <= r_ratio and r_ratio < 0.78:
            # If the average red channel value is significantly higher than the green and blue channels,
            # it's likely a penny (copper color)
            if avg_color[2] > avg_color[0] + 40 and avg_color[2] > avg_color[1] + 40:
                coin_type = "Penny"
            else:
                coin_type = "Dime"
        else:
            coin_type = "Unknown"

        results.append({"coin_type": coin_type, "position": (x, y, r)})

    return results


def draw_circles(image, circles, coin_types):
    output = image.copy()
    overlay = image.copy()
    color = (0, 255, 0)

    for (x, y, r), coin_type in zip(circles, coin_types):
        cv2.circle(output, (x, y), r, color, 2)

        # Calculate the size of the text and the background rectangle
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(coin_type, font, font_scale, 2)

        # Define the background rectangle's top-left and bottom-right points
        rect_tl = (x - r // 2, y - text_size[1] // 2 - 25)
        rect_br = (x - r // 2 + text_size[0], y + text_size[1] // 2 + 5)

        # Draw the background rectangle with a custom transparency level
        alpha = 0.3
        cv2.rectangle(overlay, rect_tl, rect_br, (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Draw the text on top of the background rectangle
        cv2.putText(output, coin_type, (x - r // 2, y), font, font_scale, color, 2)

        # Clear the overlay for the next iteration
        overlay = output.copy()

    return output


def draw_table(image, coin_counts, total_value):
    output = image.copy()
    overlay = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    text_color = (0, 255, 0)
    alpha = 0.3

    # Top-left corner of the table
    x, y = 50, 50
    row_height = 60
    col_width = 350

    # Table headers
    headers = ["Type", "Count", "Value"]

    for i, header in enumerate(headers):
        cv2.putText(
            output, header, (x + i * col_width, y), font, font_scale, text_color, 2
        )

    y += row_height

    # Draw table rows
    for coin_type, count, value in coin_counts:
        cv2.putText(output, coin_type, (x, y), font, font_scale, text_color, 2)
        cv2.putText(
            output, str(count), (x + col_width, y), font, font_scale, text_color, 2
        )
        cv2.putText(
            output,
            f"${value:.2f}",
            (x + 2 * col_width, y),
            font,
            font_scale,
            text_color,
            2,
        )
        y += row_height

    # Draw total row
    cv2.putText(output, "Total", (x, y), font, font_scale, text_color, 2)
    cv2.putText(
        output,
        str(sum(count for _, count, _ in coin_counts)),
        (x + col_width, y),
        font,
        font_scale,
        text_color,
        2,
    )
    cv2.putText(
        output,
        f"${total_value:.2f}",
        (x + 2 * col_width, y),
        font,
        font_scale,
        text_color,
        2,
    )

    # Add semi-transparent background to the table
    cv2.rectangle(overlay, (x, 0), (x + 3 * col_width, y + row_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


image_path = "coins.jpg"
results = detect_coins(image_path)

# Calculate coin counts and total value
coin_values = {"Quarter": 0.25, "Dime": 0.10, "Nickel": 0.05, "Penny": 0.01}
coin_counts = {key: 0 for key in coin_values.keys()}
for result in results:
    coin_type = result["coin_type"]
    if coin_type in coin_counts:
        coin_counts[coin_type] += 1

total_value = sum(
    coin_counts[coin_type] * coin_values[coin_type] for coin_type in coin_counts
)
coin_counts_list = [
    (coin_type, count, count * coin_values[coin_type])
    for coin_type, count in coin_counts.items()
]

# Print table to the console
print(f"{'Type':<10}{'Count':<10}{'Value':<10}")
for coin_type, count, value in coin_counts_list:
    print(f"{coin_type:<10}{count:<10}{value:<10.2f}")
print(
    f"{'Total':<10}{sum(count for _, count, _ in coin_counts_list):<10}{total_value:<10.2f}"
)

# Draw green circles on the detected coins
circles = [result["position"] for result in results]
coin_types = [result["coin_type"] for result in results]
output_image = draw_circles(cv2.imread(image_path), circles, coin_types)

# Draw the table on the output image
output_image = draw_table(output_image, coin_counts_list, total_value)

# Save the output image with green circles, coin type text, and table
cv2.imwrite("coins-out.jpg", output_image)
