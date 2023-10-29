#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { BackendStack } from '../lib/backend-stack';
import { FrontendStack } from '../lib/frontend-stack';


const app = new cdk.App();
const backendStack = new BackendStack(app, 'BackendStack', { env: { region: 'us-east-1' } });
const frontendStack = new FrontendStack(app, 'FrontendStack', {
  vpc: backendStack.backendStackProps.vpc,
  openSearchDomain: backendStack.backendStackProps.openSearchDomain,
  openSearchDomainArn: backendStack.backendStackProps.openSearchDomainArn,
  firehoseName: backendStack.backendStackProps.firehoseName,
  firehoseArn: backendStack.backendStackProps.firehoseArn,
  bucket: backendStack.backendStackProps.bucket,
  env: { region: 'us-east-1' }
});