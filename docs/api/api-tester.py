#!/usr/bin/env python3
"""
API Testing and Code Generation Tool for Prometheus AI Platform
Provides interactive API testing and generates client code in multiple languages
"""

import asyncio
import aiohttp
import json
import yaml
import argparse
import sys
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from pathlib import Path
import jinja2

@dataclass
class APITestCase:
    """Represents an API test case"""
    name: str
    description: str
    method: str
    endpoint: str
    headers: Dict[str, str]
    body: Optional[Dict[str, Any]]
    expected_status: int
    expected_response: Optional[Dict[str, Any]]
    timeout: float = 30.0

@dataclass
class TestResult:
    """Represents the result of an API test"""
    test_case: APITestCase
    success: bool
    status_code: int
    response_data: Optional[Dict[str, Any]]
    response_time: float
    error_message: Optional[str]
    timestamp: datetime

class APITester:
    """Comprehensive API testing tool with code generation capabilities"""
    
    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={'User-Agent': 'Prometheus-API-Tester/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def add_auth_header(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Add authentication header if token is available"""
        if self.auth_token:
            headers = headers.copy()
            headers['Authorization'] = f'Bearer {self.auth_token}'
        return headers
    
    async def run_test(self, test_case: APITestCase) -> TestResult:
        """Run a single API test case"""
        start_time = time.time()
        
        try:
            # Prepare request
            url = f"{self.base_url}{test_case.endpoint}"
            headers = self.add_auth_header(test_case.headers)
            
            # Add correlation ID
            headers['X-Correlation-ID'] = f"test-{int(time.time())}-{test_case.name}"
            
            # Make request
            async with self.session.request(
                method=test_case.method,
                url=url,
                headers=headers,
                json=test_case.body,
                timeout=aiohttp.ClientTimeout(total=test_case.timeout)
            ) as response:
                
                response_time = time.time() - start_time
                
                try:
                    response_data = await response.json()
                except:
                    response_data = {"raw_response": await response.text()}
                
                # Check if test passed
                success = response.status == test_case.expected_status
                
                if test_case.expected_response and success:
                    # Validate response structure
                    success = self.validate_response_structure(
                        response_data, 
                        test_case.expected_response
                    )
                
                return TestResult(
                    test_case=test_case,
                    success=success,
                    status_code=response.status,
                    response_data=response_data,
                    response_time=response_time,
                    error_message=None if success else f"Status {response.status} != {test_case.expected_status}",
                    timestamp=datetime.now()
                )
                
        except asyncio.TimeoutError:
            return TestResult(
                test_case=test_case,
                success=False,
                status_code=0,
                response_data=None,
                response_time=time.time() - start_time,
                error_message="Request timeout",
                timestamp=datetime.now()
            )
        except Exception as e:
            return TestResult(
                test_case=test_case,
                success=False,
                status_code=0,
                response_data=None,
                response_time=time.time() - start_time,
                error_message=str(e),
                timestamp=datetime.now()
            )
    
    def validate_response_structure(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Validate that response has expected structure"""
        try:
            for key, expected_value in expected.items():
                if key not in actual:
                    return False
                
                if isinstance(expected_value, dict) and isinstance(actual[key], dict):
                    if not self.validate_response_structure(actual[key], expected_value):
                        return False
                elif isinstance(expected_value, type):
                    if not isinstance(actual[key], expected_value):
                        return False
            
            return True
        except:
            return False
    
    async def run_test_suite(self, test_cases: List[APITestCase]) -> List[TestResult]:
        """Run a suite of test cases"""
        results = []
        
        print(f"Running {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Running: {test_case.name}")
            
            result = await self.run_test(test_case)
            results.append(result)
            self.test_results.append(result)
            
            # Print result
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"  {status} - {result.response_time:.2f}s - Status: {result.status_code}")
            
            if not result.success and result.error_message:
                print(f"  Error: {result.error_message}")
        
        return results
    
    def generate_test_report(self, results: List[TestResult]) -> str:
        """Generate a comprehensive test report"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        
        avg_response_time = sum(r.response_time for r in results) / total_tests if total_tests > 0 else 0
        
        report = f"""
# API Test Report - Prometheus AI Platform

**Generated:** {datetime.now().isoformat()}
**Total Tests:** {total_tests}
**Passed:** {passed_tests}
**Failed:** {failed_tests}
**Success Rate:** {(passed_tests/total_tests*100):.1f}%
**Average Response Time:** {avg_response_time:.2f}s

## Test Results

"""
        
        for result in results:
            status_icon = "✅" if result.success else "❌"
            report += f"""
### {status_icon} {result.test_case.name}

- **Method:** {result.test_case.method}
- **Endpoint:** {result.test_case.endpoint}
- **Status Code:** {result.status_code}
- **Response Time:** {result.response_time:.2f}s
- **Success:** {result.success}
"""
            
            if result.error_message:
                report += f"- **Error:** {result.error_message}\n"
            
            if result.response_data:
                report += f"""
**Response:**
```json
{json.dumps(result.response_data, indent=2)}
```
"""
        
        return report

class CodeGenerator:
    """Generates client code in multiple programming languages"""
    
    def __init__(self):
        self.templates = self.load_templates()
    
    def load_templates(self) -> Dict[str, str]:
        """Load code generation templates"""
        return {
            'python': '''
import requests
import json
from typing import Dict, Any, Optional

class PrometheusAPIClient:
    """Python client for Prometheus AI Platform API"""
    
    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session = requests.Session()
        
        if auth_token:
            self.session.headers.update({
                'Authorization': f'Bearer {auth_token}'
            })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        # Add correlation ID
        headers = kwargs.get('headers', {})
        headers['X-Correlation-ID'] = f"client-{int(time.time())}"
        kwargs['headers'] = headers
        
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    
    {% for endpoint in endpoints %}
    def {{ endpoint.method_name }}(self{% for param in endpoint.parameters %}, {{ param.name }}: {{ param.type }}{% endfor %}) -> Dict[str, Any]:
        """{{ endpoint.description }}"""
        {% if endpoint.path_params %}
        endpoint = "{{ endpoint.path }}".format({% for param in endpoint.path_params %}{{ param.name }}={{ param.name }}{% if not loop.last %}, {% endif %}{% endfor %})
        {% else %}
        endpoint = "{{ endpoint.path }}"
        {% endif %}
        
        {% if endpoint.query_params %}
        params = {
            {% for param in endpoint.query_params %}
            '{{ param.name }}': {{ param.name }},
            {% endfor %}
        }
        {% endif %}
        
        {% if endpoint.body_params %}
        data = {
            {% for param in endpoint.body_params %}
            '{{ param.name }}': {{ param.name }},
            {% endfor %}
        }
        {% endif %}
        
        response = self._make_request(
            '{{ endpoint.method }}',
            endpoint{% if endpoint.query_params %},
            params=params{% endif %}{% if endpoint.body_params %},
            json=data{% endif %}
        )
        
        return response.json()
    
    {% endfor %}

# Example usage:
if __name__ == "__main__":
    client = PrometheusAPIClient(
        base_url="https://api.prometheus-ai.com",
        auth_token="your_jwt_token_here"
    )
    
    # Example API call
    try:
        result = client.analyze_geological_data(
            data_type="seismic",
            spatial_extent={
                "min_lat": 45.0,
                "max_lat": 46.0,
                "min_lon": -122.0,
                "max_lon": -121.0
            }
        )
        print("Analysis result:", result)
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
''',
            
            'javascript': '''
/**
 * JavaScript/Node.js client for Prometheus AI Platform API
 */

class PrometheusAPIClient {
    constructor(baseUrl, authToken = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.authToken = authToken;
    }
    
    async _makeRequest(method, endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const headers = {
            'Content-Type': 'application/json',
            'X-Correlation-ID': `client-${Date.now()}`,
            ...options.headers
        };
        
        if (this.authToken) {
            headers['Authorization'] = `Bearer ${this.authToken}`;
        }
        
        const config = {
            method,
            headers,
            ...options
        };
        
        if (options.body) {
            config.body = JSON.stringify(options.body);
        }
        
        const response = await fetch(url, config);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    {% for endpoint in endpoints %}
    async {{ endpoint.method_name }}({% for param in endpoint.parameters %}{{ param.name }}{% if not loop.last %}, {% endif %}{% endfor %}) {
        /**
         * {{ endpoint.description }}
         */
        {% if endpoint.path_params %}
        const endpoint = `{{ endpoint.path }}`.replace(
            {% for param in endpoint.path_params %}
            /\\{{{ param.name }}\\}/g, {{ param.name }}{% if not loop.last %},{% endif %}
            {% endfor %}
        );
        {% else %}
        const endpoint = '{{ endpoint.path }}';
        {% endif %}
        
        const options = {};
        
        {% if endpoint.query_params %}
        const params = new URLSearchParams({
            {% for param in endpoint.query_params %}
            {{ param.name }}: {{ param.name }},
            {% endfor %}
        });
        endpoint += `?${params.toString()}`;
        {% endif %}
        
        {% if endpoint.body_params %}
        options.body = {
            {% for param in endpoint.body_params %}
            {{ param.name }}: {{ param.name }},
            {% endfor %}
        };
        {% endif %}
        
        return this._makeRequest('{{ endpoint.method }}', endpoint, options);
    }
    
    {% endfor %}
}

// Example usage:
const client = new PrometheusAPIClient(
    'https://api.prometheus-ai.com',
    'your_jwt_token_here'
);

// Example API call
client.analyzeGeologicalData({
    data_type: 'seismic',
    spatial_extent: {
        min_lat: 45.0,
        max_lat: 46.0,
        min_lon: -122.0,
        max_lon: -121.0
    }
}).then(result => {
    console.log('Analysis result:', result);
}).catch(error => {
    console.error('API error:', error);
});

module.exports = PrometheusAPIClient;
''',
            
            'curl': '''
#!/bin/bash
# cURL examples for Prometheus AI Platform API

# Set your authentication token
AUTH_TOKEN="your_jwt_token_here"
BASE_URL="https://api.prometheus-ai.com"

# Helper function for authenticated requests
api_request() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    curl -X "$method" \\
        -H "Authorization: Bearer $AUTH_TOKEN" \\
        -H "Content-Type: application/json" \\
        -H "X-Correlation-ID: curl-$(date +%s)" \\
        ${data:+-d "$data"} \\
        "$BASE_URL$endpoint"
}

{% for endpoint in endpoints %}
# {{ endpoint.description }}
{{ endpoint.method_name }}() {
    {% if endpoint.body_params %}
    local data='{{ endpoint.example_body | tojson }}'
    api_request "{{ endpoint.method }}" "{{ endpoint.path }}" "$data"
    {% else %}
    api_request "{{ endpoint.method }}" "{{ endpoint.path }}"
    {% endif %}
}

{% endfor %}

# Example usage:
echo "Testing Prometheus AI Platform API..."

# Health check
echo "Checking service health..."
api_request "GET" "/health"

# Geological analysis example
echo "Running geological analysis..."
analyze_geological_data

echo "API testing complete!"
'''
        }
    
    def generate_client_code(self, language: str, api_spec: Dict[str, Any]) -> str:
        """Generate client code for specified language"""
        
        if language not in self.templates:
            raise ValueError(f"Unsupported language: {language}")
        
        # Extract endpoints from OpenAPI spec
        endpoints = self.extract_endpoints(api_spec)
        
        # Use Jinja2 for template rendering
        template = jinja2.Template(self.templates[language])
        
        return template.render(
            endpoints=endpoints,
            api_title=api_spec.get('info', {}).get('title', 'API'),
            api_version=api_spec.get('info', {}).get('version', '1.0.0')
        )
    
    def extract_endpoints(self, api_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract endpoint information from OpenAPI spec"""
        endpoints = []
        
        for path, methods in api_spec.get('paths', {}).items():
            for method, operation in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    endpoint = {
                        'path': path,
                        'method': method.upper(),
                        'method_name': self.generate_method_name(operation.get('operationId', f"{method}_{path}")),
                        'description': operation.get('summary', ''),
                        'parameters': self.extract_parameters(operation),
                        'path_params': [],
                        'query_params': [],
                        'body_params': [],
                        'example_body': {}
                    }
                    
                    # Categorize parameters
                    for param in endpoint['parameters']:
                        if param.get('in') == 'path':
                            endpoint['path_params'].append(param)
                        elif param.get('in') == 'query':
                            endpoint['query_params'].append(param)
                    
                    # Extract request body parameters
                    if 'requestBody' in operation:
                        body_schema = operation['requestBody'].get('content', {}).get('application/json', {}).get('schema', {})
                        endpoint['body_params'] = self.extract_body_params(body_schema)
                        endpoint['example_body'] = self.generate_example_body(body_schema)
                    
                    endpoints.append(endpoint)
        
        return endpoints
    
    def generate_method_name(self, operation_id: str) -> str:
        """Generate a clean method name from operation ID"""
        # Convert to snake_case for Python, camelCase for JavaScript
        name = operation_id.replace('-', '_').replace(' ', '_')
        return ''.join(word.lower() if i == 0 else word.capitalize() for i, word in enumerate(name.split('_')))
    
    def extract_parameters(self, operation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract parameters from operation"""
        parameters = []
        
        for param in operation.get('parameters', []):
            parameters.append({
                'name': param.get('name'),
                'type': self.get_python_type(param.get('schema', {})),
                'in': param.get('in'),
                'required': param.get('required', False),
                'description': param.get('description', '')
            })
        
        return parameters
    
    def extract_body_params(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract body parameters from schema"""
        params = []
        
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        for name, prop in properties.items():
            params.append({
                'name': name,
                'type': self.get_python_type(prop),
                'required': name in required,
                'description': prop.get('description', '')
            })
        
        return params
    
    def get_python_type(self, schema: Dict[str, Any]) -> str:
        """Convert OpenAPI schema type to Python type annotation"""
        schema_type = schema.get('type', 'any')
        
        type_mapping = {
            'string': 'str',
            'integer': 'int',
            'number': 'float',
            'boolean': 'bool',
            'array': 'List[Any]',
            'object': 'Dict[str, Any]'
        }
        
        return type_mapping.get(schema_type, 'Any')
    
    def generate_example_body(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate example request body from schema"""
        example = {}
        
        properties = schema.get('properties', {})
        
        for name, prop in properties.items():
            prop_type = prop.get('type')
            
            if 'example' in prop:
                example[name] = prop['example']
            elif prop_type == 'string':
                example[name] = f"example_{name}"
            elif prop_type == 'integer':
                example[name] = 1
            elif prop_type == 'number':
                example[name] = 1.0
            elif prop_type == 'boolean':
                example[name] = True
            elif prop_type == 'array':
                example[name] = []
            elif prop_type == 'object':
                example[name] = {}
        
        return example

def create_default_test_cases() -> List[APITestCase]:
    """Create default test cases for Prometheus AI Platform"""
    
    return [
        APITestCase(
            name="Health Check",
            description="Test service health endpoint",
            method="GET",
            endpoint="/health",
            headers={"Content-Type": "application/json"},
            body=None,
            expected_status=200,
            expected_response={"status": str, "timestamp": str}
        ),
        
        APITestCase(
            name="Geological Analysis - Seismic",
            description="Test seismic data analysis",
            method="POST",
            endpoint="/analyze/geological",
            headers={"Content-Type": "application/json"},
            body={
                "data_type": "seismic",
                "spatial_extent": {
                    "min_lat": 45.0,
                    "max_lat": 46.0,
                    "min_lon": -122.0,
                    "max_lon": -121.0
                },
                "parameters": {
                    "frequency_range": [1.0, 10.0],
                    "magnitude_threshold": 2.0
                }
            },
            expected_status=200,
            expected_response={"status": str, "results": dict}
        ),
        
        APITestCase(
            name="Physics Simulation - Thermal",
            description="Test thermal physics simulation",
            method="POST",
            endpoint="/simulate/physics",
            headers={"Content-Type": "application/json"},
            body={
                "simulation_type": "thermal",
                "initial_conditions": {
                    "temperature": 1200.0,
                    "pressure": 200e6
                },
                "boundary_conditions": {
                    "surface_temperature": 15.0
                },
                "time_span": 86400.0
            },
            expected_status=200,
            expected_response={"status": str, "results": dict}
        )
    ]

async def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description="Prometheus AI Platform API Tester")
    parser.add_argument("--base-url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--auth-token", help="JWT authentication token")
    parser.add_argument("--generate-code", choices=['python', 'javascript', 'curl'], help="Generate client code")
    parser.add_argument("--spec-file", help="OpenAPI specification file")
    parser.add_argument("--output-file", help="Output file for generated code")
    parser.add_argument("--run-tests", action="store_true", help="Run API tests")
    
    args = parser.parse_args()
    
    if args.generate_code:
        if not args.spec_file:
            print("Error: --spec-file required for code generation")
            sys.exit(1)
        
        # Load OpenAPI spec
        with open(args.spec_file, 'r') as f:
            if args.spec_file.endswith('.yaml') or args.spec_file.endswith('.yml'):
                spec = yaml.safe_load(f)
            else:
                spec = json.load(f)
        
        # Generate code
        generator = CodeGenerator()
        code = generator.generate_client_code(args.generate_code, spec)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(code)
            print(f"Generated {args.generate_code} client code: {args.output_file}")
        else:
            print(code)
    
    elif args.run_tests:
        # Run API tests
        async with APITester(args.base_url, args.auth_token) as tester:
            test_cases = create_default_test_cases()
            results = await tester.run_test_suite(test_cases)
            
            # Generate report
            report = tester.generate_test_report(results)
            
            # Save report
            report_file = f"api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\nTest report saved: {report_file}")
            
            # Exit with error code if tests failed
            failed_count = sum(1 for r in results if not r.success)
            if failed_count > 0:
                print(f"\n{failed_count} tests failed!")
                sys.exit(1)
            else:
                print("\nAll tests passed!")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())