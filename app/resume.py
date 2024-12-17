import uuid
import chromadb
import json
from sentence_transformers import SentenceTransformer


class Resume:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="resume")

    def load_resume(self):
        # Define your resume sections
        resume_sections = {

            "Technical Skills": """Programming Languages: C, C#, Java, Python, HTML5, CSS, JavaScript, TypeScript, JSON
        Frameworks & Technologies: .NET (WinForms, WPF with MVVM, ASP.NET MVC Core), Entity Framework, ADO.NET, Angular, React, Node.js, Razor, Bootstrap v5, Dependency Injection, Swagger, SSIS, SSRS, PowerShell, Visual Studio Extensibility, Spring Boot, Agile/Scrum, CI/CD
        Databases & Machine Learning: SQL, PL/SQL, Microsoft SQL Server, MongoDB, PostgreSQL, Scikit-learn, Pandas
        Cloud Platforms: Azure, AWS (S3, Lambda, EC2, RDS), GCP
        Tools & Version Control: Jenkins, JIRA, TortoiseSVN, Git, Bitbucket, Postman, Visual Studio, IntelliJ, Harness, OpenShift
        Certifications: Microsoft Certified: Azure Fundamentals""",

            "Professional Experience": """Student Assistant, REGARDS Study, UAB, Birmingham, Alabama August 2023 – April 2024
        Developed a patient records management system in ASP.NET Core Web App, enhancing data accuracy and reliability by 30% through real-time updates based on patient profiles, medical histories, survey responses, and real-time health monitoring data.
        Designed reusable UI components (Partial Views, Layouts, Razor CRUD), reducing development time by 30% and improving platform-wide UI consistency.
        Architected Data Access Layers using ADO.NET and Entity Framework, reducing data retrieval times by 35%. Managed backend services (RESTful, SOAP, WCF and Web APIs), boosting data exchange efficiency by 30%.
        Software Engineer – UNISYS India Pvt Ltd., India April 2021 – December 2022
        Constructed Web APIs with ASP.NET Core 6.0 for a banking alert system, processing over 1 million alerts daily via OpenShift, ensuring seamless customer notifications.
        Optimized Microsoft SQL Server with efficient Data Access Layers (ADO.NET, Entity Framework), reducing query times by 40% for better database performance.
        Engineered React JS components (JSX, props, hooks), boosting user engagement by 30% and improving load times by 25% through an interactive user interface.
        Utilized Swagger and Postman for API documentation and testing, reducing API error rates by 20%.
        Enhanced a custom domain-driven microservices architecture by implementing CQRS and MediatR design patterns, resulting in a 30% improvement in system scalability and maintainability.
        Worked in Agile/Scrum using JIRA for project management and GitHub for code management. Implemented GitHub Actions and Harness in the CI/CD pipeline, boosting deployment efficiency by 35% on OpenShift.
        Associate Software Engineer – UNISYS India Pvt Ltd., India June 2019 – April 2021
        Augmented MCP Project with Visual Studio extensibility features, improving user experience by 75%.
        Designed WPF, Custom Controls, User Controls and WinForms using MVVM patterns, boosting UI responsiveness by 30% and user satisfaction by 20%.
        Led the development and deployment of a custom debug engine integrated with the MCP Program, revolutionizing debugging capabilities for specialized languages such as Algol and Cobol files.
        Contributed significantly (50%) to the Azure Network Watcher research group for MCP, specializing in enhancing network health for MCP's IaaS products, including Virtual Machines, Servers, Virtual Networks, Application Gateways, and Load Balancers.
        Created continuous integration and delivery pipelines using Jenkins, reducing deployment time by 40% and increasing deployment frequency by 30%. Managed diverse Agile/Scrum assignments, resolved 200+ bugs, and employed Tortoise SVN for version control.""",

            "Project Work": """Event Management App: Web app for customers to explore event services; used React, NodeJS, MongoDB.
        IPL Result Prediction: Predicted IPL tournament results with data models; used Python, Flask, Pandas.
        Human Activity Mining: Analyzed routines to provide health care suggestions; achieved 92% accuracy using RNN."""
        }

        # Add sections to the collection with embeddings
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Embed the resume sections and add them to the collection
        for section, content in resume_sections.items():
            section_embedding = embedder.encode([content])
            self.collection.add(
                documents=[content],
                metadatas=[{"section": section}],
                embeddings=section_embedding,
                ids=[str(uuid.uuid4())]
            )


    def query_links(self, res):
        # Add sections to the collection with embeddings
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        #return self.collection.query(query_texts=skills, n_results=2).get('metadatas', [])
        # Your job description as a JSON object
        job_description_json = res

        # Convert job description to text (you can modify the structure to match your query requirements)
        job_description_text = json.dumps(job_description_json)

        # Generate embeddings for the job description
        job_description_embedding = embedder.encode([job_description_text])

        # Query the collection to find matching sections based on the job description
        search_results = self.collection.query(
            query_embeddings=job_description_embedding,  # Pass the embeddings of the job description
            n_results=5  # Fetch top 5 results
        )

        # Extract the relevant "links" from the matching results
        matching_links = []
        for result in search_results['documents']:
            matching_links.append(result)
        return matching_links
