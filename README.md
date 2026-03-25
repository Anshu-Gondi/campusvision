# Face Recognition Attendance System

## Project Overview
This project is a comprehensive Face Recognition Attendance System designed to automate the attendance process using advanced facial recognition technology.

## Features
- **Automated Attendance**: Automatically marks attendance based on face recognition.
- **User-Friendly Interface**: Simple and intuitive interface for users.
- **Real-time Processing**: Quick processing time for efficient attendance marking.
- **Multiple Recognition Modes**: Supports single and group attendance marking.

## Architecture Details
The system employs a microservices architecture consisting of several components:
- **Frontend**: Built with HTML/CSS and JavaScript for user interaction.
- **Backend**: RESTful API developed with Node.js and Express.
- **Database**: Utilizes MongoDB for storing user and attendance data.
- **Face Recognition Module**: Integrated with OpenCV for real-time face detection and recognition.

## Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Anshu-Gondi/campusvision.git
   cd campusvision
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Set up the database:
   - Create a MongoDB database and update the connection string in the `.env` file.
4. Start the server:
   ```bash
   npm start
   ```

## Usage Instructions
- To mark attendance, navigate to the attendance page and follow the on-screen instructions.
- Ensure the camera is enabled for real-time recognition.

## Contribution Guidelines
We welcome contributions! Please fork the repository and submit a pull request. Ensure to follow the existing code style and include relevant tests.

## License Information
This project is licensed under the MIT License. See the LICENSE file for more details.