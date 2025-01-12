pipeline {
    agent any

    environment {
        AWS_ACCESS_KEY_ID = credentials('aws-account-access') // Replace with Jenkins credentials ID for AWS
        AWS_SECRET_ACCESS_KEY = credentials('aws-account-access') // Replace with Jenkins credentials ID for AWS
    }

    stages {
        stage('Checkout Code') {
            steps {
                echo 'Cloning the GitHub repository...'
                // Explicit Git clone command
                sh 'git clone https://github.com/USRUTUJA/MLops-Project-Repo.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                sh 'docker build -t mlops-train-model:latest .'
            }
        }

        stage('Run Docker Container') {
            steps {
                echo 'Running Docker container...'
                sh '''
                docker run --rm \
                    -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
                    -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
                    mlops-train-model:latest
                '''
            }
        }

        stage('Clean Up Docker') {
            steps {
                echo 'Cleaning up unused Docker resources...'
                sh '''
                docker system prune -f
                '''
            }
        }
    }

    post {
        success {
            echo 'Pipeline executed successfully.'
        }
        failure {
            echo 'Pipeline failed. Check the logs for more details.'
        }
    }
}
