from neo4j import GraphDatabase
import spacy

class GraphDBExecutor():
    def __init__(self,neo4j_uri,neo4j_user, neo4j_password):

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))