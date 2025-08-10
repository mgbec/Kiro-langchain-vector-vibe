#!/usr/bin/env python3
"""
AWS Resource Setup Script
Creates necessary AWS resources for vector database storage.
"""

import boto3
import json
import time
from typing import Optional

class AWSResourceSetup:
    def __init__(self, region_name="us-east-1"):
        self.region = region_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.iam_client = boto3.client('iam', region_name=region_name)
        self.opensearch_client = boto3.client('opensearch', region_name=region_name)
        
    def create_s3_bucket(self, bucket_name: str) -> bool:
        """Create S3 bucket for vector database storage."""
        try:
            if self.region == 'us-east-1':
                # us-east-1 doesn't need LocationConstraint
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            # Enable versioning
            self.s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            print(f"✅ Created S3 bucket: {bucket_name}")
            return True
            
        except self.s3_client.exceptions.BucketAlreadyExists:
            print(f"⚠️  S3 bucket {bucket_name} already exists")
            return True
        except Exception as e:
            print(f"❌ Failed to create S3 bucket: {e}")
            return False
    
    def create_bedrock_execution_role(self) -> Optional[str]:
        """Create IAM role for Bedrock Knowledge Base."""
        role_name = "AmazonBedrockExecutionRoleForKnowledgeBase"
        
        # Trust policy for Bedrock
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # Permissions policy
        permissions_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        "arn:aws:s3:::*",
                        "arn:aws:s3:::*/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "aoss:APIAccessAll"
                    ],
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        try:
            # Create role
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Execution role for Bedrock Knowledge Base"
            )
            
            role_arn = response['Role']['Arn']
            
            # Attach inline policy
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName="BedrockKnowledgeBasePolicy",
                PolicyDocument=json.dumps(permissions_policy)
            )
            
            print(f"✅ Created IAM role: {role_arn}")
            return role_arn
            
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            # Role already exists, get its ARN
            response = self.iam_client.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            print(f"⚠️  IAM role already exists: {role_arn}")
            return role_arn
            
        except Exception as e:
            print(f"❌ Failed to create IAM role: {e}")
            return None
    
    def create_opensearch_domain(self, domain_name: str) -> Optional[str]:
        """Create OpenSearch domain for vector search."""
        try:
            response = self.opensearch_client.create_domain(
                DomainName=domain_name,
                EngineVersion='OpenSearch_2.3',
                ClusterConfig={
                    'InstanceType': 't3.small.search',
                    'InstanceCount': 1,
                    'DedicatedMasterEnabled': False
                },
                EBSOptions={
                    'EBSEnabled': True,
                    'VolumeType': 'gp3',
                    'VolumeSize': 10
                },
                AccessPolicies=json.dumps({
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "AWS": "*"
                            },
                            "Action": "es:*",
                            "Resource": f"arn:aws:es:{self.region}:*:domain/{domain_name}/*"
                        }
                    ]
                }),
                DomainEndpointOptions={
                    'EnforceHTTPS': True
                },
                NodeToNodeEncryptionOptions={
                    'Enabled': True
                },
                EncryptionAtRestOptions={
                    'Enabled': True
                }
            )
            
            domain_arn = response['DomainStatus']['ARN']
            endpoint = response['DomainStatus'].get('Endpoint')
            
            print(f"✅ Creating OpenSearch domain: {domain_name}")
            print(f"   ARN: {domain_arn}")
            print("   ⏳ Domain creation takes 10-15 minutes...")
            
            return domain_arn
            
        except Exception as e:
            print(f"❌ Failed to create OpenSearch domain: {e}")
            return None
    
    def setup_complete_environment(self, 
                                 bucket_name: str,
                                 opensearch_domain: Optional[str] = None):
        """Set up complete AWS environment for vector databases."""
        print("Setting up AWS resources for vector databases...")
        print("=" * 50)
        
        # Create S3 bucket
        s3_success = self.create_s3_bucket(bucket_name)
        
        # Create Bedrock execution role
        bedrock_role = self.create_bedrock_execution_role()
        
        # Optionally create OpenSearch domain
        opensearch_arn = None
        if opensearch_domain:
            opensearch_arn = self.create_opensearch_domain(opensearch_domain)
        
        print("\n" + "=" * 50)
        print("Setup Summary:")
        print(f"✅ S3 Bucket: {bucket_name}" if s3_success else f"❌ S3 Bucket: {bucket_name}")
        print(f"✅ Bedrock Role: {bedrock_role}" if bedrock_role else "❌ Bedrock Role: Failed")
        if opensearch_domain:
            print(f"✅ OpenSearch Domain: {opensearch_arn}" if opensearch_arn else f"❌ OpenSearch Domain: Failed")
        
        print("\nNext Steps:")
        print("1. Update your .env file with the created resource names")
        print("2. Wait for OpenSearch domain to be active (if created)")
        print("3. Run your vector database examples")
        
        return {
            "s3_bucket": bucket_name if s3_success else None,
            "bedrock_role": bedrock_role,
            "opensearch_domain": opensearch_arn
        }


def main():
    print("AWS Vector Database Resource Setup")
    print("=" * 40)
    
    # Configuration
    bucket_name = input("Enter S3 bucket name for vector storage: ").strip()
    if not bucket_name:
        print("❌ Bucket name is required")
        return
    
    create_opensearch = input("Create OpenSearch domain? (y/n): ").strip().lower() == 'y'
    opensearch_domain = None
    if create_opensearch:
        opensearch_domain = input("Enter OpenSearch domain name: ").strip()
        if not opensearch_domain:
            print("❌ OpenSearch domain name is required")
            return
    
    # Setup resources
    setup = AWSResourceSetup()
    results = setup.setup_complete_environment(bucket_name, opensearch_domain)
    
    # Generate .env content
    print("\n" + "=" * 50)
    print("Add these to your .env file:")
    print(f"S3_BUCKET_NAME={bucket_name}")
    if opensearch_domain and results.get('opensearch_domain'):
        print(f"OPENSEARCH_URL=https://{opensearch_domain}.{setup.region}.es.amazonaws.com")


if __name__ == "__main__":
    main()