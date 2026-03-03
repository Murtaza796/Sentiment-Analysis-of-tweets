pipeline {
    agent any

    tools {
        maven 'M3'   // Must match name in Jenkins Global Tool Config
        jdk 'JDK-17'        // Must match your configured JDK name
    }

    stages {

        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Clean') {
            steps {
                sh 'mvn clean'
            }
        }

        stage('Compile') {
            steps {
                sh 'mvn compile'
            }
        }

    }

    post {
        success {
            echo 'Build Successful!'
        }
        failure {
            echo 'Build Failed!'
        }
    }
}
