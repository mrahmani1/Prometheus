/**
 * Academic API Service
 * Integrates with SemanticScholar, ConnectedPapers, WileyOnline, and GeoscienceWorld APIs
 * for causal analysis and hypothesis forming modules
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';

export interface AcademicPaper {
  id: string;
  title: string;
  authors: Array<{
    name: string;
    affiliations: string[];
  }>;
  abstract: string;
  year: number;
  venue: string;
  doi?: string;
  url: string;
  citationCount: number;
  influentialCitationCount: number;
  keywords: string[];
  fieldsOfStudy: string[];
  references: string[];
  citations: string[];
  relevanceScore: number;
  causalRelationships?: Array<{
    source: string;
    target: string;
    relationship: string;
    confidence: number;
  }>;
}

export interface CausalHypothesis {
  id: string;
  hypothesis: string;
  variables: string[];
  expectedRelationships: Array<{
    source: string;
    target: string;
    type: 'causal' | 'correlational' | 'mediating';
    strength: number;
  }>;
  supportingEvidence: AcademicPaper[];
  contradictingEvidence: AcademicPaper[];
  confidence: number;
  domain: string;
}

export interface LiteratureSearchQuery {
  query: string;
  fields?: string[];
  yearRange?: { start: number; end: number };
  venue?: string[];
  authors?: string[];
  limit?: number;
  offset?: number;
  sortBy?: 'relevance' | 'citationCount' | 'year' | 'influentialCitationCount';
}

export interface CausalExtractionResult {
  relationships: Array<{
    source: string;
    target: string;
    relationship: string;
    confidence: number;
    evidence: string;
    paperIds: string[];
  }>;
  variables: Array<{
    name: string;
    type: string;
    domain: string;
    frequency: number;
  }>;
  hypotheses: CausalHypothesis[];
}

class AcademicAPIService {
  private static instance: AcademicAPIService;
  private semanticScholarClient: AxiosInstance;
  private connectedPapersClient: AxiosInstance;
  private wileyClient: AxiosInstance;
  private geoscienceWorldClient: AxiosInstance;

  private constructor() {
    // Initialize clients with default values
    this.semanticScholarClient = axios.create();
    this.connectedPapersClient = axios.create();
    this.wileyClient = axios.create();
    this.geoscienceWorldClient = axios.create();
    
    this.initializeClients();
  }

  public static getInstance(): AcademicAPIService {
    if (!AcademicAPIService.instance) {
      AcademicAPIService.instance = new AcademicAPIService();
    }
    return AcademicAPIService.instance;
  }

  private initializeClients(): void {
    // Semantic Scholar API client
    this.semanticScholarClient = axios.create({
      baseURL: 'https://api.semanticscholar.org/graph/v1',
      timeout: 30000,
      headers: {
        'Accept': 'application/json',
        'x-api-key': process.env.REACT_APP_SEMANTIC_SCHOLAR_API_KEY || '',
      },
    });

    // Connected Papers API client
    this.connectedPapersClient = axios.create({
      baseURL: 'https://api.connectedpapers.com/v2',
      timeout: 30000,
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${process.env.REACT_APP_CONNECTED_PAPERS_API_KEY || ''}`,
      },
    });

    // Wiley Online Library API client
    this.wileyClient = axios.create({
      baseURL: 'https://api.wiley.com/onlinelibrary/tdm/v1',
      timeout: 30000,
      headers: {
        'Accept': 'application/json',
        'Wiley-TDM-Client-Token': process.env.REACT_APP_WILEY_API_KEY || '',
      },
    });

    // Geoscience World API client
    this.geoscienceWorldClient = axios.create({
      baseURL: 'https://api.geoscienceworld.org/v1',
      timeout: 30000,
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${process.env.REACT_APP_GEOSCIENCE_WORLD_API_KEY || ''}`,
      },
    });

    // Add error interceptors
    [this.semanticScholarClient, this.connectedPapersClient, this.wileyClient, this.geoscienceWorldClient].forEach(client => {
      client.interceptors.response.use(
        response => response,
        error => {
          console.error('Academic API error:', error);
          return Promise.reject(error);
        }
      );
    });
  }

  /**
   * Search academic literature across all sources
   */
  async searchLiterature(query: LiteratureSearchQuery): Promise<AcademicPaper[]> {
    const results: AcademicPaper[] = [];

    try {
      // Search Semantic Scholar
      const semanticResults = await this.searchSemanticScholar(query);
      results.push(...semanticResults);

      // Search Wiley Online Library
      const wileyResults = await this.searchWiley(query);
      results.push(...wileyResults);

      // Search Geoscience World
      const geoscienceResults = await this.searchGeoscienceWorld(query);
      results.push(...geoscienceResults);

      // Remove duplicates and sort by relevance
      const uniqueResults = this.deduplicatePapers(results);
      return uniqueResults.sort((a, b) => b.relevanceScore - a.relevanceScore);
    } catch (error) {
      console.error('Literature search failed:', error);
      throw error;
    }
  }

  /**
   * Search Semantic Scholar API
   */
  private async searchSemanticScholar(query: LiteratureSearchQuery): Promise<AcademicPaper[]> {
    try {
      const params: any = {
        query: query.query,
        limit: query.limit || 100,
        offset: query.offset || 0,
        fields: 'paperId,title,authors,abstract,year,venue,citationCount,influentialCitationCount,references,citations,fieldsOfStudy,url',
      };

      if (query.yearRange) {
        params.year = `${query.yearRange.start}-${query.yearRange.end}`;
      }

      if (query.fields && query.fields.length > 0) {
        params.fieldsOfStudy = query.fields.join(',');
      }

      const response: AxiosResponse = await this.semanticScholarClient.get('/paper/search', { params });

      return response.data.data.map((paper: any) => ({
        id: paper.paperId,
        title: paper.title,
        authors: paper.authors?.map((author: any) => ({
          name: author.name,
          affiliations: author.affiliations || [],
        })) || [],
        abstract: paper.abstract || '',
        year: paper.year || 0,
        venue: paper.venue || '',
        doi: paper.externalIds?.DOI,
        url: paper.url || `https://www.semanticscholar.org/paper/${paper.paperId}`,
        citationCount: paper.citationCount || 0,
        influentialCitationCount: paper.influentialCitationCount || 0,
        keywords: [],
        fieldsOfStudy: paper.fieldsOfStudy || [],
        references: paper.references?.map((ref: any) => ref.paperId) || [],
        citations: paper.citations?.map((cit: any) => cit.paperId) || [],
        relevanceScore: this.calculateRelevanceScore(paper, query),
      }));
    } catch (error) {
      console.error('Semantic Scholar search failed:', error);
      return [];
    }
  }

  /**
   * Search Wiley Online Library
   */
  private async searchWiley(query: LiteratureSearchQuery): Promise<AcademicPaper[]> {
    try {
      const params: any = {
        query: query.query,
        count: query.limit || 50,
        start: query.offset || 1,
      };

      if (query.yearRange) {
        params.publicationDateFrom = `${query.yearRange.start}-01-01`;
        params.publicationDateTo = `${query.yearRange.end}-12-31`;
      }

      const response: AxiosResponse = await this.wileyClient.get('/articles', { params });

      return response.data.result?.map((article: any) => ({
        id: article.identifier?.doi || article.identifier?.value,
        title: article.title,
        authors: article.authors?.map((author: any) => ({
          name: `${author.givenNames} ${author.surname}`,
          affiliations: author.affiliations || [],
        })) || [],
        abstract: article.abstract || '',
        year: new Date(article.publicationDate).getFullYear(),
        venue: article.publicationTitle,
        doi: article.identifier?.doi,
        url: article.onlineReadingUrl || article.pdfUrl,
        citationCount: 0, // Wiley doesn't provide citation counts
        influentialCitationCount: 0,
        keywords: article.keywords || [],
        fieldsOfStudy: article.subjects || [],
        references: [],
        citations: [],
        relevanceScore: this.calculateRelevanceScore(article, query),
      })) || [];
    } catch (error) {
      console.error('Wiley search failed:', error);
      return [];
    }
  }

  /**
   * Search Geoscience World
   */
  private async searchGeoscienceWorld(query: LiteratureSearchQuery): Promise<AcademicPaper[]> {
    try {
      const params: any = {
        q: query.query,
        rows: query.limit || 50,
        start: query.offset || 0,
      };

      if (query.yearRange) {
        params.fq = `publication_date:[${query.yearRange.start}-01-01T00:00:00Z TO ${query.yearRange.end}-12-31T23:59:59Z]`;
      }

      const response: AxiosResponse = await this.geoscienceWorldClient.get('/search', { params });

      return response.data.response?.docs?.map((doc: any) => ({
        id: doc.id,
        title: doc.title_display,
        authors: doc.author_display?.map((author: string) => ({
          name: author,
          affiliations: [],
        })) || [],
        abstract: doc.abstract || '',
        year: doc.publication_date ? new Date(doc.publication_date).getFullYear() : 0,
        venue: doc.journal_title || doc.book_title,
        doi: doc.doi,
        url: doc.url,
        citationCount: 0,
        influentialCitationCount: 0,
        keywords: doc.keywords || [],
        fieldsOfStudy: doc.subject || [],
        references: [],
        citations: [],
        relevanceScore: this.calculateRelevanceScore(doc, query),
      })) || [];
    } catch (error) {
      console.error('Geoscience World search failed:', error);
      return [];
    }
  }

  /**
   * Get connected papers for a given paper
   */
  async getConnectedPapers(paperId: string): Promise<AcademicPaper[]> {
    try {
      const response: AxiosResponse = await this.connectedPapersClient.get(`/graph/${paperId}`);
      
      return response.data.papers?.map((paper: any) => ({
        id: paper.id,
        title: paper.title,
        authors: paper.authors?.map((author: any) => ({
          name: author.name,
          affiliations: author.affiliations || [],
        })) || [],
        abstract: paper.abstract || '',
        year: paper.year || 0,
        venue: paper.venue || '',
        doi: paper.doi,
        url: paper.url,
        citationCount: paper.citationCount || 0,
        influentialCitationCount: paper.influentialCitationCount || 0,
        keywords: paper.keywords || [],
        fieldsOfStudy: paper.fieldsOfStudy || [],
        references: [],
        citations: [],
        relevanceScore: paper.similarity || 0,
      })) || [];
    } catch (error) {
      console.error('Connected Papers API failed:', error);
      return [];
    }
  }

  /**
   * Extract causal relationships from literature
   */
  async extractCausalRelationships(papers: AcademicPaper[], domain: string = 'geology'): Promise<CausalExtractionResult> {
    try {
      // Use NLP techniques to extract causal relationships from abstracts and titles
      const relationships: Array<{
        source: string;
        target: string;
        relationship: string;
        confidence: number;
        evidence: string;
        paperIds: string[];
      }> = [];

      const variables: Map<string, { name: string; type: string; domain: string; frequency: number }> = new Map();
      const hypotheses: CausalHypothesis[] = [];

      // Causal keywords and patterns
      const causalPatterns = [
        /(\w+)\s+(?:causes?|leads? to|results? in|triggers?|induces?|produces?)\s+(\w+)/gi,
        /(\w+)\s+(?:is caused by|results from|is triggered by|is induced by)\s+(\w+)/gi,
        /(?:due to|because of|as a result of)\s+(\w+),?\s+(\w+)/gi,
        /(\w+)\s+(?:influences?|affects?|controls?|determines?)\s+(\w+)/gi,
      ];

      for (const paper of papers) {
        const text = `${paper.title} ${paper.abstract}`;
        
        // Extract causal relationships using pattern matching
        for (const pattern of causalPatterns) {
          let match;
          while ((match = pattern.exec(text)) !== null) {
            const source = this.normalizeVariable(match[1]);
            const target = this.normalizeVariable(match[2]);
            
            if (source && target && source !== target) {
              relationships.push({
                source,
                target,
                relationship: 'causal',
                confidence: this.calculateCausalConfidence(match[0], paper),
                evidence: match[0],
                paperIds: [paper.id],
              });

              // Track variables
              [source, target].forEach(variable => {
                if (variables.has(variable)) {
                  variables.get(variable)!.frequency++;
                } else {
                  variables.set(variable, {
                    name: variable,
                    type: this.inferVariableType(variable, domain),
                    domain,
                    frequency: 1,
                  });
                }
              });
            }
          }
        }
      }

      // Generate hypotheses based on extracted relationships
      const relationshipGroups = this.groupRelationshipsByVariables(relationships);
      for (const [variablePair, rels] of relationshipGroups) {
        if (rels.length >= 3) { // Require multiple papers supporting the relationship
          const avgConfidence = rels.reduce((sum, rel) => sum + rel.confidence, 0) / rels.length;
          const supportingPapers = papers.filter(p => rels.some(r => r.paperIds.includes(p.id)));
          
          hypotheses.push({
            id: `hyp_${variablePair}`,
            hypothesis: `${rels[0].source} has a causal influence on ${rels[0].target}`,
            variables: [rels[0].source, rels[0].target],
            expectedRelationships: [{
              source: rels[0].source,
              target: rels[0].target,
              type: 'causal',
              strength: avgConfidence,
            }],
            supportingEvidence: supportingPapers,
            contradictingEvidence: [],
            confidence: avgConfidence,
            domain,
          });
        }
      }

      return {
        relationships: this.consolidateRelationships(relationships),
        variables: Array.from(variables.values()),
        hypotheses,
      };
    } catch (error) {
      console.error('Causal relationship extraction failed:', error);
      throw error;
    }
  }

  /**
   * Generate research hypotheses based on literature gaps
   */
  async generateResearchHypotheses(
    topic: string,
    existingKnowledge: CausalExtractionResult,
    targetVariables: string[]
  ): Promise<CausalHypothesis[]> {
    try {
      // Search for recent literature on the topic
      const recentPapers = await this.searchLiterature({
        query: topic,
        yearRange: { start: 2020, end: new Date().getFullYear() },
        limit: 200,
      });

      // Extract relationships from recent literature
      const recentRelationships = await this.extractCausalRelationships(recentPapers, 'geology');

      // Identify gaps and potential new hypotheses
      const hypotheses: CausalHypothesis[] = [];

      // Look for unexplored variable combinations
      for (const sourceVar of targetVariables) {
        for (const targetVar of targetVariables) {
          if (sourceVar === targetVar) continue;

          const existingRelation = existingKnowledge.relationships.find(
            r => r.source === sourceVar && r.target === targetVar
          );

          const recentRelation = recentRelationships.relationships.find(
            r => r.source === sourceVar && r.target === targetVar
          );

          // If no existing relationship but recent evidence suggests one
          if (!existingRelation && recentRelation && recentRelation.confidence > 0.6) {
            hypotheses.push({
              id: `gap_hyp_${sourceVar}_${targetVar}`,
              hypothesis: `${sourceVar} may have a previously unrecognized causal influence on ${targetVar}`,
              variables: [sourceVar, targetVar],
              expectedRelationships: [{
                source: sourceVar,
                target: targetVar,
                type: 'causal',
                strength: recentRelation.confidence,
              }],
              supportingEvidence: recentPapers.filter(p => recentRelation.paperIds.includes(p.id)),
              contradictingEvidence: [],
              confidence: recentRelation.confidence * 0.8, // Reduce confidence for gap hypotheses
              domain: 'geology',
            });
          }
        }
      }

      return hypotheses;
    } catch (error) {
      console.error('Hypothesis generation failed:', error);
      throw error;
    }
  }

  /**
   * Validate causal hypothesis against literature
   */
  async validateHypothesis(hypothesis: CausalHypothesis): Promise<{
    validation: 'supported' | 'contradicted' | 'inconclusive';
    supportingPapers: AcademicPaper[];
    contradictingPapers: AcademicPaper[];
    confidence: number;
    recommendations: string[];
  }> {
    try {
      // Search for papers related to the hypothesis
      const searchQuery = `${hypothesis.variables.join(' ')} causal relationship ${hypothesis.domain}`;
      const papers = await this.searchLiterature({
        query: searchQuery,
        limit: 100,
      });

      // Analyze papers for support or contradiction
      const supportingPapers: AcademicPaper[] = [];
      const contradictingPapers: AcademicPaper[] = [];

      for (const paper of papers) {
        const text = `${paper.title} ${paper.abstract}`.toLowerCase();
        const variables = hypothesis.variables.map(v => v.toLowerCase());
        
        // Check if paper mentions the variables
        const mentionsVariables = variables.every(v => text.includes(v));
        if (!mentionsVariables) continue;

        // Look for supporting or contradicting evidence
        const supportingPatterns = [
          new RegExp(`${variables[0]}.*(?:causes?|leads? to|results? in).*${variables[1]}`, 'i'),
          new RegExp(`${variables[1]}.*(?:is caused by|results from).*${variables[0]}`, 'i'),
        ];

        const contradictingPatterns = [
          new RegExp(`${variables[0]}.*(?:does not cause|has no effect on).*${variables[1]}`, 'i'),
          new RegExp(`no.*(?:relationship|correlation|association).*${variables[0]}.*${variables[1]}`, 'i'),
        ];

        const hasSupport = supportingPatterns.some(pattern => pattern.test(text));
        const hasContradiction = contradictingPatterns.some(pattern => pattern.test(text));

        if (hasSupport && !hasContradiction) {
          supportingPapers.push(paper);
        } else if (hasContradiction && !hasSupport) {
          contradictingPapers.push(paper);
        }
      }

      // Determine validation result
      const supportRatio = supportingPapers.length / (supportingPapers.length + contradictingPapers.length);
      let validation: 'supported' | 'contradicted' | 'inconclusive';
      let confidence: number;

      if (supportRatio > 0.7) {
        validation = 'supported';
        confidence = supportRatio;
      } else if (supportRatio < 0.3) {
        validation = 'contradicted';
        confidence = 1 - supportRatio;
      } else {
        validation = 'inconclusive';
        confidence = 0.5;
      }

      // Generate recommendations
      const recommendations: string[] = [];
      if (validation === 'inconclusive') {
        recommendations.push('Conduct additional empirical studies to clarify the relationship');
        recommendations.push('Consider mediating variables that might explain conflicting results');
      } else if (validation === 'supported') {
        recommendations.push('Investigate the mechanisms underlying this causal relationship');
        recommendations.push('Test the relationship under different conditions');
      } else {
        recommendations.push('Reconsider the hypothesis or look for alternative explanations');
        recommendations.push('Examine potential confounding variables');
      }

      return {
        validation,
        supportingPapers,
        contradictingPapers,
        confidence,
        recommendations,
      };
    } catch (error) {
      console.error('Hypothesis validation failed:', error);
      throw error;
    }
  }

  // Helper methods

  private calculateRelevanceScore(paper: any, query: LiteratureSearchQuery): number {
    let score = 0;
    const title = (paper.title || '').toLowerCase();
    const abstract = (paper.abstract || '').toLowerCase();
    const queryTerms = query.query.toLowerCase().split(' ');

    // Title matches
    queryTerms.forEach(term => {
      if (title.includes(term)) score += 2;
      if (abstract.includes(term)) score += 1;
    });

    // Citation count bonus
    if (paper.citationCount) {
      score += Math.log(paper.citationCount + 1) * 0.1;
    }

    // Recent publication bonus
    if (paper.year && paper.year > 2015) {
      score += (paper.year - 2015) * 0.1;
    }

    return score;
  }

  private deduplicatePapers(papers: AcademicPaper[]): AcademicPaper[] {
    const seen = new Set<string>();
    return papers.filter(paper => {
      const key = paper.doi || paper.title.toLowerCase().replace(/\s+/g, '');
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  private normalizeVariable(variable: string): string {
    return variable.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .trim()
      .replace(/\s+/g, '_');
  }

  private calculateCausalConfidence(evidence: string, paper: AcademicPaper): number {
    let confidence = 0.5;

    // Strong causal language
    if (/causes?|results? in|leads? to/i.test(evidence)) confidence += 0.3;
    if (/triggers?|induces?|produces?/i.test(evidence)) confidence += 0.2;
    if (/influences?|affects?/i.test(evidence)) confidence += 0.1;

    // Paper quality indicators
    if (paper.citationCount > 50) confidence += 0.1;
    if (paper.influentialCitationCount > 10) confidence += 0.1;
    if (paper.year > 2015) confidence += 0.05;

    return Math.min(confidence, 1.0);
  }

  private inferVariableType(variable: string, _domain: string): string {
    const geologicalTerms = {
      'temperature': 'thermal',
      'pressure': 'mechanical',
      'stress': 'mechanical',
      'strain': 'mechanical',
      'deformation': 'mechanical',
      'seismic': 'seismic',
      'earthquake': 'seismic',
      'volcanic': 'volcanic',
      'eruption': 'volcanic',
      'magma': 'geochemical',
      'mineral': 'geochemical',
      'chemical': 'geochemical',
    };

    for (const [term, type] of Object.entries(geologicalTerms)) {
      if (variable.includes(term)) return type;
    }

    return 'unknown';
  }

  private groupRelationshipsByVariables(relationships: Array<{
    source: string;
    target: string;
    relationship: string;
    confidence: number;
    evidence: string;
    paperIds: string[];
  }>): Map<string, typeof relationships> {
    const groups = new Map<string, typeof relationships>();

    for (const rel of relationships) {
      const key = `${rel.source}_${rel.target}`;
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key)!.push(rel);
    }

    return groups;
  }

  private consolidateRelationships(relationships: Array<{
    source: string;
    target: string;
    relationship: string;
    confidence: number;
    evidence: string;
    paperIds: string[];
  }>): Array<{
    source: string;
    target: string;
    relationship: string;
    confidence: number;
    evidence: string;
    paperIds: string[];
  }> {
    const consolidated = new Map<string, {
      source: string;
      target: string;
      relationship: string;
      confidence: number;
      evidence: string;
      paperIds: string[];
    }>();

    for (const rel of relationships) {
      const key = `${rel.source}_${rel.target}`;
      if (consolidated.has(key)) {
        const existing = consolidated.get(key)!;
        existing.confidence = Math.max(existing.confidence, rel.confidence);
        existing.paperIds = [...new Set([...existing.paperIds, ...rel.paperIds])];
        existing.evidence += `; ${rel.evidence}`;
      } else {
        consolidated.set(key, { ...rel });
      }
    }

    return Array.from(consolidated.values());
  }
}

export default AcademicAPIService;