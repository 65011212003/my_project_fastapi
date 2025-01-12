import requests
import json
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import spacy
from collections import Counter
import time
import logging
from datetime import datetime
import openpyxl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://localhost:8000"
OUTPUT_DIR = "test_results"
SAVE_RESULTS = True

class RiceDiseaseAnalysisTester:
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "api_tests": {},
            "data_quality": {},
            "entity_extraction": {},
            "translation": {}
        }

    def test_api_endpoints(self):
        """Test all API endpoints"""
        logger.info("Starting API endpoint tests...")
        
        endpoints_to_test = {
            "status": {"method": "GET", "endpoint": "/status"},
            "analyze": {
                "method": "POST",
                "endpoint": "/analyze",
                "data": {
                    "query": "rice blast disease",
                    "max_results": 100
                }
            },
            "clusters": {"method": "GET", "endpoint": "/clusters"},
            "statistics": {"method": "GET", "endpoint": "/statistics"},
            "research_locations": {"method": "GET", "endpoint": "/research-locations"}
        }

        results = {}
        for name, config in endpoints_to_test.items():
            try:
                start_time = time.time()
                
                if config["method"] == "GET":
                    response = requests.get(f"{BASE_URL}{config['endpoint']}")
                else:  # POST
                    response = requests.post(
                        f"{BASE_URL}{config['endpoint']}", 
                        json=config.get("data", {})
                    )
                
                response_time = time.time() - start_time
                
                results[name] = {
                    "status": "PASS" if response.status_code == 200 else "FAIL",
                    "response_time": f"{response_time:.2f}s",
                    "status_code": response.status_code
                }
                
                logger.info(f"{name} endpoint: {results[name]['status']}")
                
            except Exception as e:
                results[name] = {
                    "status": "FAIL",
                    "error": str(e)
                }
                logger.error(f"Error testing {name} endpoint: {str(e)}")

        self.test_results["api_tests"] = results
        return results

    def test_data_quality(self):
        """Test quality of processed data"""
        logger.info("Starting data quality tests...")
        
        try:
            df = pd.read_excel("rice_disease_analysis/rice_disease_pubmed_data.xlsx", engine='openpyxl')
            
            # Missing data analysis
            missing_data = df.isnull().sum()
            missing_percentage = (missing_data / len(df)) * 100
            
            # Cluster analysis
            cluster_metrics = {}
            if 'Cluster' in df.columns:
                unique_clusters = df['Cluster'].nunique()
                cluster_dist = df['Cluster'].value_counts().to_dict()
                cluster_metrics = {
                    "unique_clusters": unique_clusters,
                    "cluster_distribution": cluster_dist
                }
            
            # Text processing quality
            df['abstract_length'] = df['Processed_Abstract'].str.len()
            text_quality = df['abstract_length'].describe().to_dict()
            
            results = {
                "total_records": len(df),
                "missing_data_percentage": missing_percentage.to_dict(),
                "cluster_metrics": cluster_metrics,
                "text_processing_quality": text_quality
            }
            
            self.test_results["data_quality"] = results
            logger.info("Data quality tests completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"Error in data quality testing: {str(e)}"
            logger.error(error_msg)
            results = {"error": error_msg, "total_records": 0}
            self.test_results["data_quality"] = results
            return results

    def test_entity_extraction(self):
        """Test entity extraction quality"""
        logger.info("Starting entity extraction tests...")
        
        try:
            df = pd.read_excel("rice_disease_analysis/rice_disease_pubmed_data.xlsx", engine='openpyxl')
            
            # Entity analysis
            all_entities = []
            entity_categories = set()
            successful_extractions = 0
            
            for entities_str in df['Entities']:
                try:
                    entities_dict = eval(entities_str)
                    successful_extractions += 1
                    
                    for category, items in entities_dict.items():
                        entity_categories.add(category)
                        all_entities.extend(items)
                except:
                    continue
            
            entity_counts = Counter(all_entities)
            
            results = {
                "total_documents": len(df),
                "successful_extractions": successful_extractions,
                "extraction_success_rate": f"{(successful_extractions/len(df))*100:.2f}%",
                "unique_entities": len(set(all_entities)),
                "entity_categories": list(entity_categories),
                "top_entities": dict(entity_counts.most_common(10))
            }
            
            self.test_results["entity_extraction"] = results
            logger.info("Entity extraction tests completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"Error in entity extraction testing: {str(e)}"
            logger.error(error_msg)
            self.test_results["entity_extraction"] = {"error": error_msg}
            return {"error": error_msg}

    def test_translation(self):
        """Test translation functionality"""
        logger.info("Starting translation tests...")
        
        test_texts = {
            "single": "Rice blast disease is a significant problem in Asian rice cultivation.",
            "bulk": [
                "Rice blast disease",
                "Bacterial leaf blight",
                "Brown spot disease"
            ]
        }
        
        results = {"single": {}, "bulk": {}}
        
        # Test single translation
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/translate",
                json={"text": test_texts["single"], "target_language": "th"}
            )
            response_time = time.time() - start_time
            
            results["single"] = {
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "response_time": f"{response_time:.2f}s",
                "translated_text": response.json().get("translated_text") if response.status_code == 200 else None
            }
            
        except Exception as e:
            results["single"] = {"status": "FAIL", "error": str(e)}
        
        # Test bulk translation
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/translate-bulk",
                json={"texts": test_texts["bulk"], "target_language": "th"}
            )
            response_time = time.time() - start_time
            
            results["bulk"] = {
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "response_time": f"{response_time:.2f}s",
                "translations": response.json().get("translations") if response.status_code == 200 else None
            }
            
        except Exception as e:
            results["bulk"] = {"status": "FAIL", "error": str(e)}
        
        self.test_results["translation"] = results
        return results

    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        report = {
            "test_summary": {
                "timestamp": self.test_results["timestamp"],
                "overall_status": "PASS"
            },
            "detailed_results": self.test_results
        }
        
        # Calculate overall status
        failed_tests = []
        
        # Check API tests
        for endpoint, result in self.test_results.get("api_tests", {}).items():
            if result.get("status") != "PASS":
                failed_tests.append(f"API Endpoint: {endpoint}")
        
        # Check data quality
        if "error" in self.test_results.get("data_quality", {}):
            failed_tests.append("Data Quality")
        
        # Check entity extraction
        if "error" in self.test_results.get("entity_extraction", {}):
            failed_tests.append("Entity Extraction")
        
        # Check translation
        for test_type, result in self.test_results.get("translation", {}).items():
            if result.get("status") != "PASS":
                failed_tests.append(f"Translation: {test_type}")
        
        if failed_tests:
            report["test_summary"]["overall_status"] = "FAIL"
            report["test_summary"]["failed_tests"] = failed_tests
        
        # Save report to file if required
        if SAVE_RESULTS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.xlsx"
            
            # Convert report to DataFrame and save as Excel
            df = pd.DataFrame.from_dict(report, orient='index')
            df.to_excel(filename, engine='openpyxl')
            logger.info(f"Test report saved to {filename}")
        
        return report

    def print_report(self, report):
        """Print formatted test report"""
        print("\nRice Disease Analysis System Test Report")
        print("======================================")
        print(f"Timestamp: {report['test_summary']['timestamp']}")
        print(f"Overall Status: {report['test_summary']['overall_status']}")
        
        if report['test_summary']['overall_status'] == "FAIL":
            print("\nFailed Tests:")
            for test in report['test_summary']['failed_tests']:
                print(f"- {test}")
        
        print("\nDetailed Results:")
        print("-----------------")
        
        # Print API test results
        print("\nAPI Tests:")
        for endpoint, result in report['detailed_results']['api_tests'].items():
            print(f"  {endpoint}:")
            for key, value in result.items():
                print(f"    {key}: {value}")
        
        # Print data quality results
        print("\nData Quality Metrics:")
        dq_results = report['detailed_results']['data_quality']
        if "error" not in dq_results:
            print(f"  Total Records: {dq_results.get('total_records', 0)}")
            print("  Cluster Metrics:")
            print(f"    Unique Clusters: {dq_results['cluster_metrics']['unique_clusters']}")
        
        # Print entity extraction results
        print("\nEntity Extraction Results:")
        ee_results = report['detailed_results']['entity_extraction']
        if "error" not in ee_results:
            print(f"  Success Rate: {ee_results.get('extraction_success_rate', 'N/A')}")
            print(f"  Unique Entities: {ee_results.get('unique_entities', 0)}")
        
        # Print translation results
        print("\nTranslation Test Results:")
        for test_type, result in report['detailed_results']['translation'].items():
            print(f"  {test_type}:")
            print(f"    Status: {result['status']}")
            print(f"    Response Time: {result.get('response_time', 'N/A')}")

def run_all_tests():
    """Run all tests and generate report"""
    tester = RiceDiseaseAnalysisTester()
    
    # Run all tests
    tester.test_api_endpoints()
    tester.test_data_quality()
    tester.test_entity_extraction()
    tester.test_translation()
    
    # Generate and print report
    report = tester.generate_test_report()
    tester.print_report(report)

if __name__ == "__main__":
    run_all_tests()