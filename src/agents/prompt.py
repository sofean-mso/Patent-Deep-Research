

def get_patent_review_prompt()-> str:
    return ("""
    You are an expert research assistant tasked with writing a comprehensive technical scientific report for '''{topic}''' based on a provided collection of patent documents. 
                Each document includes a patent number and the associated summary. 
                The report should be detailed, formal, and structured according to the following guidelines:\n

            🔹 Report Structure:\n
                - Abstract (Summary):\n
                    Provide a concise summary of the inventions covered the topic {topic} in the provided patent texts.\n

                - Explain the background context and existing technologies or challenges the inventions address related.\n

                - Technical Fields of Invention:\n
                    Provide a detailed list of technological areas and fields of invention that related to the topic {topic}, provide them with citations.'\n

              - Inventions related to the topic/question  '''{topic}'''
                    - show with details a list of most related inventions with novelty and objectives.\n
                    - Highlight unique components, devices, apparatus, methods, or systems.\n
                    - Identify the technical problems solved and how the invention provides an improvement.\n

              - Applicability and Uses:\n
                 Discuss with details practical applications and uses of the most inventions related to the topic {topic}.\n

              - Conclusion:\n
                    Summarize the overall inventions of the patents.\n  

            🔹 Formatting and Style:
                Use formal, technical language appropriate for a research or patent analyst audience.\n      
                Reference each patent by its number (e.g., "as described in US1234567").\n
                Group similar or related inventions where appropriate to avoid redundancy.\n
                use markdown format. \n
                Where useful, include tables, bullet points, or diagrams (optional, based on capabilities).
        - Requirements:
            - All information in the report should relate to {topic} \n
            - Don't provide information outside the topic '''{topic}'''\n
            - Ensure that all generated content is specifically focused on the complete topic — for example, ‘cold plasma for wound healing’ — rather than discussing ‘cold plasma’ or ‘wound healing’ in isolation. The report information should clearly address how cold plasma is used within the context of wound healing, covering aspects such as underlying mechanisms, therapeutic benefits, practical applications, and supporting research.
            - Use patent numbers as citations when discussing specific inventions like (US 20180307744). This citation is provided along with the contexts, and don't provide citations outside the provided contexts.\n
            - Do not hallucinate.\n
            - Do not include any irrelevant information. \n

            PATENT SUMMARIES: '''{patent_research_results}''' \n
            at the end of your report, provide a list of citations that only used in the report.
    """)

def get_patent_answer_prompt()-> str:
    return ("""
        You will be provided with patent passages as context to answer the question at the end. Please follow the following rules:
        Output Format:
             - Format your response as a JSON object with these exact keys:
             - "answer": the answer of the question
             - "sources": list of sources               
            
              Example 1:
                '''{{
                    "answer": "CAP kills cancer cells by triggering the rise of intracellular reactive oxygen species (ROS), DNA damage, mitochondrial damage, or cellular membrane damage (US20190231411A1). The rise of intracellular ROS always occurs in cancer cells upon CAP treatment, which causes a noticeable damage on the antioxidant system and subsequently DNA double strands break (DSB) to a fatal degree (US10479979B2). Serious DNA damage and other effect of CAP on cancer cells result in the cell cycle arrest, apoptosis or necrosis with a dose-dependent pattern (US10479979B2).. ...."
                    "sources": ["US20190231411A1", "US10479979B2"]
                }}'''
              Example 2:
                '''{{
                    "answer": "some medical and cosmetic applications are used for treating non-malignant skin growths, nail fungus infections, skin rejuvenation, and wrinkle removal. ...."
                    "sources": ["WO2022178164A1", "US20220256682A1", "US10479979B2", "US20220256682A1"]
                }}'''  
                 
        Requirements:
            - keep the answer concise.
            - ALWAYS return answer with a "sources" part in your answer.
            - For each part of your answer, indicate which sources most support it via valid citation markers at the end of sentences, like (US20220168565A1).
            - The "sources" is provided with each passage in the context like "US10023858B2", and do not change it. 
            - Use only the context to answer the question. 
            - If the answer is not in the provided context or passages, just say 'No Answer Found!
        
        Context:\n {patent_research_results}\n
        Question: \n{research_topic}?\n
        Answer:
    """)


def get_patent_reflection_prompt()-> str:
    return ("""
     You are an expert research assistant analyzing patent summaries about the question or research topic '''{research_topic}'''
                
                Instructions:
                    - Detect knowledge gaps or areas requiring further exploration, and generate only one follow-up query.
                    - If the provided patent summaries fully address the user’s question, do not create additional queries.
                    - When gaps exist, formulate a follow-up query that deepen or broaden understanding.

                 Requirements:
                    - Ensure the follow-up query is self-contained and includes necessary concepts for patent search.
                    - provide only one query and should take the form of a natural-language sentence.
                    - Do not include any irrelevant information outside the topic/question '''{research_topic}'''.
                    - Keep it concise.
                    - Do not hallucinate.
                    
            Output Format:
                - Format your response as a JSON object with these exact keys:
                - "is_sufficient": true or false 
                - "knowledge_gap": Describe what information is missing or needs clarification 
                - "follow_up_query": Write a specific question to address this gap.
                
            Example:
            '''json
            {{
                "is_sufficient": true, or false
                "knowledge_gap": "The patent summary lacks information about examples of core inventions",  "" if is_sufficient is true
                "follow_up_query": "What are core inventions for [specific technology]?", "" if is_sufficient is true
            }}
            ''' 
            Reflect carefully on the patent Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output in JSON format:   
            PATENT SUMMARIES: '''{patent_research_results}''' \n
    """)

def get_rerank_prompt_template()-> str:
    return ("""
     You are an assistant whose role is to evaluate how relevant a given patent document or passage is to a specified topic.
        You will be provided with a Topic and a Patent Document/passage.\n
        Your task is to assign a relevance score from 0 to 5, and below is your grading rubric: 
             0 = not relevant at all. 
             5 = highly relevant. \n
             
        - Your relevance score should reflect whether the document addresses the entire topic.
        For example, if the topic is “cold plasma for skin treatment”, the document must relate to both “cold plasma” and “skin treatment” in order to be considered relevant.
            
        Instructions:
            - Read the topic and the patent document carefully.
            - Determine how clearly the document relates to the given topic.
            - Do not make assumptions beyond the provided text.
            - Respond only with a single score between 0 and 5.

        Topic:  '''{topic}'''\n
        Document: '''{doc}''' \n
        Score:
    """)


def get_summary_prompt_template()-> str:
    return ( """
    You will be provided with patent context including: Title, Abstract, Description, and Claims of a patent document.\n
        From only the provided patent texts, write a concise summary which includes: \n
            - the technical field or area of the invention, \n
            - the objectives of the invention,\n
            - the uses or applications of the invention, and,\n
            - the core of the invention or the novelty extracted from the first sentence of the claims.\n
            - the summary of the abstract, technical effects, technical problems, and technical means.\n

        Instructions:
            - Keep it concise.
            - Do not hallucinate.
            - Do not include any irrelevant information.
            
        context:  '''{context}'''\n
        CONCISE SUMMARY:
    """)

def get_patent_queries_prompt_template() -> str:
    return ( """
    Your task is to construct a sophisticated patent search queries for a provided question. 
         The queries should take the form of a natural-language sentence, similar to how concepts are expressed in patent documents, 
         and it should describe the given research question in detail. 
         The response should capture both the broad technical scope and the specific features of the subject, ensuring alignment with patent-related terminology and context.
               
                - Requirements:
                - Each query should focus on one specific aspect of the provided question.
                - Each query should include all concepts in the provided question.
                - Don't produce more than 3 queries.
                - Don't generate multiple similar queries, 1 is enough.
                - Do not include any irrelevant or speculative information.
                
                Format: 
                - Format your response as a JSON object with ALL two of these exact keys:
                   - "rationale": Brief explanation of why these queries are relevant
                   - "query": A list of search queries
                
                Example:
                
                question: What are apparatus, devices, and methods that are used for plasma nail surface treatment?
                ```json
                {{
                    "rationale": "To answer this comparative growth question accurately, we need specific concepts. These queries target the precise scientific information needed: device name, methods, apparatus, technological concepts",
                    "query": ["plasma apparatus, instruments, and devices are commonly employed in plasma-based treatments for modifying or functionalizing nail surfaces, nail treatment,, including both cosmetic and biomedical applications", "methods, protocols, and treatment techniques used for applying cold plasma to nail surfaces for purposes such as improving adhesion, sterilization, or cleaning", "cold plasma, atmospheric plasma, or low-pressure plasma systems, are used for nail surfaces"],
                }}
                ```
                question: {research_topic}
    """)

def get_patent_query_prompt_template() -> str:
    return ("""
        Your task is to construct a sophisticated patent search query. 
             Requirements:
                    - Ensure the query query is self-contained and includes necessary concepts for patent search.
                    - provide the query in the form of a natural-language sentence.
                    - Do not include any irrelevant information to the topic/question '''{research_topic}'''.
                    - Your response should encompass all key terms and concepts found in the texts for the given topic.
                    - Keep it concise.
                    - Do not hallucinate.

             <EXAMPLE>
                "research_topic": "cold plasma for skin treatment",
                "query": "cold plasma technology for skin treatment, focusing on its therapeutic effects for wound healing, acne, skin rejuvenation, and other dermatological applications."
            </EXAMPLE>
            <EXAMPLE>
                "research_question": "How is plasma used for nail surface treatment?",
                "query": "use of plasma in nail surface modification, fungal infection treatment, nails cleaning, nail care, and nail surface."
            </EXAMPLE>

            <TOPIC>
            {research_topic}
            </TOPIC>

            Return only the query text.
    """)