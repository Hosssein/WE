/*==========================================================================
 *
 *  Original source copyright (c) 2001, Carnegie Mellon University.
 *  See copyright.cmu for details.
 *  Modifications copyright (c) 2002, University of Massachusetts.
 *  See copyright.umass for details.
 *
 *==========================================================================
 */



#include "RetMethod.h"
#include "Param.hpp"
#include "common_headers.hpp"
#include <cmath>
#include "DocUnigramCounter.hpp"
#include "RelDocUnigramCounter.hpp"
#include "IndexManager.hpp"
#include "OneStepMarkovChain.hpp"
#include <vector>
#include <set>
#include <algorithm>
#include <FreqVector.hpp>

#include <fstream>
#include <sstream>


using namespace lemur::api;
using namespace lemur::retrieval;
using namespace lemur::utility;

bool pairCompare(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem);

extern int RSMethodHM; // 0--> LM , 1--> RecSys
extern int negGenModeHM;//0 --> coll , 1--> nonRel
extern int feedbackMode;
extern int updatingThresholdMode; // 0 ->no updating ,1->linear

//extern map<int, map<int,double> > FEEDBACKMAP;
static int qid=1;
static string RM;
Index* myIndex = NULL;
void lemur::retrieval::QueryModel::interpolateWith(const lemur::langmod::UnigramLM &qModel,
                                                   double origModCoeff,
                                                   int howManyWord,
                                                   double prSumThresh,
                                                   double prThresh) {
    if (!qm) {
        qm = new lemur::api::IndexedRealVector();
    } else {
        qm->clear();
    }

    qModel.startIteration();
    while (qModel.hasMore()) {
        IndexedReal entry;
        qModel.nextWordProb((TERMID_T &)entry.ind,entry.val);
        qm->push_back(entry);

    }
    qm->Sort();

    double countSum = totalCount();

    // discounting the original model
    startIteration();
    while (hasMore()) {
        QueryTerm *qt = nextTerm();
        incCount(qt->id(), qt->weight()*origModCoeff/countSum);
        delete qt;
    }

    // now adding the new model
    double prSum = 0;
    int wdCount = 0;
    IndexedRealVector::iterator it;
    it = qm->begin();
    while (it != qm->end() && prSum < prSumThresh &&
           wdCount < howManyWord && (*it).val >=prThresh) {
        incCount((*it).ind, (*it).val*(1-origModCoeff));
        prSum += (*it).val;
        it++;
        wdCount++;
    }

    //Sum w in Q qtf * log(qtcf/termcount);
    colQLikelihood = 0;
    colQueryLikelihood();
    colKLComputed = false;
}

void lemur::retrieval::QueryModel::load(istream &is)
{
    // clear existing counts
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        setCount(qt->id(),0);
    }
    colQLikelihood = 0;

    int count;
    is >> count;
    char wd[500];
    double pr;
    while (count-- >0) {
        is >> wd >> pr;
        TERMID_T id = ind.term(wd);
        if (id != 0) setCount(id, pr); // don't load OOV terms
    }
    colQueryLikelihood();
    colKLComputed = false;
}

void lemur::retrieval::QueryModel::save(ostream &os)
{
    int count = 0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        delete qt;
    }
    os << " " << count << endl;
    startIteration();
    while (hasMore()) {
        qt = nextTerm();
        os << ind.term(qt->id()) << " "<< qt->weight() << endl;
        delete qt;
    }
}

void lemur::retrieval::QueryModel::clarity(ostream &os)
{
    int count = 0;
    double sum=0, ln_Pr=0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        // query-clarity = SUM_w{P(w|Q)*log(P(w|Q)/P(w))}
        // P(w)=cf(w)/|C|
        double pw = ((double)ind.termCount(qt->id())/(double)ind.termCount());
        // P(w|Q) is a prob computed by any model, e.g. relevance models
        double pwq = qt->weight();
        sum += pwq;
        ln_Pr += (pwq)*log(pwq/pw);
        delete qt;
    }
    // clarity should be computed with log_2, so divide by log(2).
    os << "=" << count << " " << (ln_Pr/(sum ? sum : 1.0)/log(2.0)) << endl;
    startIteration();
    while (hasMore()) {
        qt = nextTerm();
        // print clarity for each query term
        // clarity should be computed with log_2, so divide by log(2).
        os << ind.term(qt->id()) << " "
           << (qt->weight()*log(qt->weight()/
                                ((double)ind.termCount(qt->id())/
                                 (double)ind.termCount())))/log(2.0) << endl;
        delete qt;
    }
}

double lemur::retrieval::QueryModel::clarity() const
{
    int count = 0;
    double sum=0, ln_Pr=0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        // query-clarity = SUM_w{P(w|Q)*log(P(w|Q)/P(w))}
        // P(w)=cf(w)/|C|
        double pw = ((double)ind.termCount(qt->id())/(double)ind.termCount());
        // P(w|Q) is a prob computed by any model, e.g. relevance models
        double pwq = qt->weight();
        sum += pwq;
        ln_Pr += (pwq)*log(pwq/pw);
        delete qt;
    }
    // normalize by sum of probabilities in the input model
    ln_Pr = ln_Pr/(sum ? sum : 1.0);
    // clarity should be computed with log_2, so divide by log(2).
    return (ln_Pr/log(2.0));
}

lemur::retrieval::RetMethod::RetMethod(const Index &dbIndex,
                                       const string &supportFileName,
                                       ScoreAccumulator &accumulator) :
    TextQueryRetMethod(dbIndex, accumulator), supportFile(supportFileName) {

    //docParam.smthMethod = RetParameter::defaultSmoothMethod;
    docParam.smthMethod = RetParameter::DIRICHLETPRIOR;
    //docParam.smthMethod = RetParameter::ABSOLUTEDISCOUNT;
    //docParam.smthMethod = RetParameter::JELINEKMERCER;

    docParam.smthStrategy= RetParameter::defaultSmoothStrategy;
    //docParam.ADDelta = RetParameter::defaultADDelta;
    docParam.JMLambda = RetParameter::defaultJMLambda;
    //docParam.JMLambda = 0.9;
    docParam.DirPrior = dbIndex.docLengthAvg();//50;//RetParameter::defaultDirPrior;

    qryParam.adjScoreMethod = RetParameter::NEGATIVEKLD;
    //qryParam.adjScoreMethod = RetParameter::QUERYLIKELIHOOD;
    //qryParam.fbMethod = RetParameter::defaultFBMethod;
    //qryParam.fbMethod = RetParameter::DIVMIN;




    qryParam.fbMethod = RetParameter::MIXTURE;
    RM="MIX";// *** Query Likelihood adjusted score method *** //
    //qryParam.fbCoeff = RetParameter::defaultFBCoeff;
    qryParam.fbCoeff =0.5;
    qryParam.fbPrTh = RetParameter::defaultFBPrTh;
    qryParam.fbPrSumTh = RetParameter::defaultFBPrSumTh;
    qryParam.fbTermCount = 15;//RetParameter::defaultFBTermCount;
    qryParam.fbMixtureNoise = RetParameter::defaultFBMixNoise;
    qryParam.emIterations = 50;//RetParameter::defaultEMIterations;

    docProbMass = NULL;
    uniqueTermCount = NULL;
    mcNorm = NULL;

    NegMu = ind.docLengthAvg();
    collectLMCounter = new lemur::langmod::DocUnigramCounter(ind);
    collectLM = new lemur::langmod::MLUnigramLM(*collectLMCounter, ind.termLexiconID());

    delta = 0.007;

    newNonRelRecieved = false;
    newRelRecieved = false;
    newNonRelRecievedCnt = 0,newRelRecievedCnt =0;

    //int W2VecSize =100;
    /*coefMatrix = new double*[W2VecSize];
    for(int i = 0 ; i< W2VecSize ; i++)
        coefMatrix[i] = new double[W2VecSize];
    for(int i = 0 ; i< W2VecSize ; i++)
        for(int j = 0 ; j < W2VecSize ; j++)
            coefMatrix[i][j]=((rand()%2000)-1000)/1000.0f;
    */
    W2VecDimSize = 100;
    coefMatrix.resize(W2VecDimSize);
    for(int i = 0 ;i < W2VecDimSize ; i++)
        coefMatrix[i].resize(W2VecDimSize);

    for(int i = 0 ;i < W2VecDimSize ; i++)
        for(int j = 0 ; j < W2VecDimSize ; j++)
            coefMatrix[i][j] = ((rand()%2000)-1000)/1000.0f;

    alphaCoef = 0.8;
    lambdaCoef = 0.05;
    betaCoef = 0.01;
    etaCoef = 1;

    queryAvgVec.assign(W2VecDimSize , -10.0);




    switch (RSMethodHM)
    {
    case 0://lm
        // setThreshold(-4.3);
        setThreshold(-6.3);

        break;
    case 1://negGen
    {
        if(negGenModeHM==0)//col//mu=2500
            // setThreshold(2.1);
            setThreshold(1.5);
        else if(negGenModeHM == 1)
            // setThreshold(2.4);
            setThreshold(1.6);
        break;
    }
    }

    //prev_distQuery = new double[ind.termCountUnique()+1];
    scFunc = new ScoreFunc();
    scFunc->setScoreMethod(qryParam.adjScoreMethod);
}

lemur::retrieval::RetMethod::~RetMethod()
{
    //delete [] prev_distQuery;
    delete [] docProbMass;
    delete [] uniqueTermCount;
    delete [] mcNorm;
    delete collectLM;
    delete collectLMCounter;
    delete scFunc;
    delete [] relComputed;
    delete [] nonRelComputed;
    //delete [] coefMatrix[]; //FIX ME!!!!
}

void lemur::retrieval::RetMethod::loadSupportFile() {
    ifstream ifs;
    int i;

    // Only need to load this file if smooth strategy is back off
    // or the smooth method is absolute discount. Don't reload if
    // docProbMass is not NULL.

    if (docProbMass == NULL &&
            (docParam.smthMethod == RetParameter::ABSOLUTEDISCOUNT ||
             docParam.smthStrategy == RetParameter::BACKOFF)) {
        cerr << "lemur::retrieval::SimpleKLRetMethod::loadSupportFile loading "
             << supportFile << endl;

        ifs.open(supportFile.c_str());
        if (ifs.fail()) {
            throw  Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                             "smoothing support file open failure");
        }
        COUNT_T numDocs = ind.docCount();
        docProbMass = new double[numDocs+1];
        uniqueTermCount = new COUNT_T[numDocs+1];
        for (i = 1; i <= numDocs; i++) {
            DOCID_T id;
            int uniqCount;
            double prMass;
            ifs >> id >> uniqCount >> prMass;
            if (id != i) {
                throw  Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                                 "alignment error in smooth support file, wrong id:");
            }
            docProbMass[i] = prMass;
            uniqueTermCount[i] = uniqCount;
        }
        ifs.close();
    }

    // only need to load this file if the feedback method is
    // markov chain. Don't reload if called a second time.

    if (mcNorm == NULL && qryParam.fbMethod == RetParameter::MARKOVCHAIN) {
        string mcSuppFN = supportFile + ".mc";
        cerr << "lemur::retrieval::SimpleKLRetMethod::loadSupportFile loading " << mcSuppFN << endl;

        ifs.open(mcSuppFN.c_str());
        if (ifs.fail()) {
            throw Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                            "Markov chain support file can't be opened");
        }

        mcNorm = new double[ind.termCountUnique()+1];

        for (i = 1; i <= ind.termCountUnique(); i++) {
            TERMID_T id;
            double norm;
            ifs >> id >> norm;
            if (id != i) {
                throw Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                                "alignment error in Markov chain support file, wrong id:");
            }
            mcNorm[i] = norm;
        }
    }
}

DocumentRep *lemur::retrieval::RetMethod::computeDocRep(DOCID_T docID)
{
    switch (docParam.smthMethod) {
    case RetParameter::JELINEKMERCER:
        return( new JMDocModel(docID,
                               ind.docLength(docID),
                               *collectLM,
                               docProbMass,
                               docParam.JMLambda,
                               docParam.smthStrategy));
    case RetParameter::DIRICHLETPRIOR:
        return (new DPriorDocModel(docID,
                                   ind.docLength(docID),
                                   *collectLM,
                                   docProbMass,
                                   docParam.DirPrior,
                                   docParam.smthStrategy));
    case RetParameter::ABSOLUTEDISCOUNT:
        return (new ABSDiscountDocModel(docID,
                                        ind.docLength(docID),
                                        *collectLM,
                                        docProbMass,
                                        uniqueTermCount,
                                        docParam.ADDelta,
                                        docParam.smthStrategy));
    case RetParameter::TWOSTAGE:
        return (new TStageDocModel(docID,
                                   ind.docLength(docID),
                                   *collectLM,
                                   docProbMass,
                                   docParam.DirPrior, // 1st stage mu
                                   docParam.JMLambda, // 2nd stage lambda
                                   docParam.smthStrategy));


    default:
        // this should throw, not exit.
        cerr << "Unknown document language model smoothing method\n";
        exit(1);
    }
}


void lemur::retrieval::RetMethod::updateProfile(lemur::api::TextQueryRep &origRep,
                                                vector<int> relJudgDoc ,vector<int> nonRelJudgDoc)
{
    cerr<<"relJudgDoc size: "<<relJudgDoc.size()<<" nonRelJudgDoc size: "<<nonRelJudgDoc.size()<<endl;

    //updateTextQuery(origRep,*relDocs,*nonRelDocs);

    if(nonRelComputed[nonRelJudgDoc.size()] == false)
    {
        computeRelNonRelDist(origRep,nonRelJudgDoc,nonRelJudgDoc,false);
        nonRelComputed[nonRelJudgDoc.size()] = true;
    }else  if(relComputed[relJudgDoc.size()] == false)
    {
        computeRelNonRelDist(origRep,relJudgDoc,relJudgDoc,true);
        relComputed[relJudgDoc.size()] = true;
    }

}
void lemur::retrieval::RetMethod::updateThreshold(lemur::api::TextQueryRep &origRep,
                                                  vector<int> relJudgDoc ,vector<int> nonReljudgDoc , int mode,double relSumScores ,double nonRelSumScore)
{
    thresholdUpdatingMethod = updatingThresholdMode;
    //double alpha = 0.3,beta = 0.9;
    if(thresholdUpdatingMethod == 0)//no updating
        return;
    else if(thresholdUpdatingMethod == 1)//linear
    {
        if(mode == 0)//non rel passed
        {
            setThreshold(getThreshold()+getC1());
            //cout<<"mode 0 "<<getThreshold()<<endl;
        }
        else if(mode == 1)//not showed anything
        {
            setThreshold( getThreshold()- getC2() );
            //cout<<"mode 1 "<<getThreshold()<<endl;
        }

        //threshold = -4.5;
    }else if (thresholdUpdatingMethod == 2)//diff rel nonrel method
    {
        double alpha = getDiffThrUpdatingParam();
        double relSize = relJudgDoc.size();
        double nonRelSize = nonReljudgDoc.size();
        double val =  alpha * std::max( ((relSumScores/(relSize+1)+0.005) - (nonRelSumScore/(nonRelSize+1)+0.005)) ,-3.5) *
                (std::abs(std::log10( (nonRelSize+1) / (relSize+1) ) + 0.005 ) );


        if(mode == 0)
            setThreshold(getThreshold() + val);
        else
            setThreshold(getThreshold() - val);
        //cout<<relSumScores<<" "<<relSize<<endl;
        //cout<<alpha<<" "<<(relSumScores/(relSize+1.0))<<" "<<(nonRelSumScore/nonRelSize+1)<<" "<<(std::log10( (nonRelSize+1) / (relSize+1) ) + 0.005 )<<endl;
        cout <<"mode "<<mode<<" alpha "<<alpha <<" relSum: "<<(relSumScores/(relSize+1)+0.005)<<" nonRelSum: "<< (nonRelSumScore/(nonRelSize+1)+0.005) <<" val: "<<val<<" log: "<<std::log10( (nonRelSize+1) / (relSize+1) );
        cout<<" thr: "<<getThreshold()<<endl;
    }


}

extern map<int,vector<double> >wordEmbedding;
float lemur::retrieval::RetMethod::cosineSim(vector<double> aa, vector<double> bb)
{
    double numerator = 0 ,denominator =0,sum_a = 0,sum_b = 0;
    for(int i = 0 ; i< aa.size() ; i++)
    {
        double a = aa[i];
        double b = bb[i];
        numerator += a*b;
        sum_a += a*a;
        sum_b += b*b;
    }
    denominator = sqrt(sum_a) * sqrt(sum_b);
    //cout<< numerator/denominator;

    return numerator /denominator ;

}

float lemur::retrieval::RetMethod::computeProfDocSim(lemur::api::TextQueryRep *textQR,int docID ,
                                                     vector<int> relJudgDoc ,vector<int> nonReljudgDoc , bool newNonRel , bool newRel)
{

#if 1
    vector<vector<double> > queryTerms,docTerms;

    double counter =0 ;
    textQR->startIteration();
    while(textQR->hasMore())
    {
        counter += 1;
        QueryTerm *qt = textQR->nextTerm();
        if(wordEmbedding.find(qt->id()) != wordEmbedding.end())
            queryTerms.push_back(wordEmbedding[qt->id()]);
        else
            continue;

        delete qt;
    }

    vector<double> queryAvg( W2VecDimSize);//FIXME!!!!!!!!!!!!!!!!!!!!!!

    for(int i =0 ; i< queryTerms.size() ; i++)
    {
        for(int j = 0 ;j<queryTerms[i].size() ; j++)
            queryAvg[j] += queryTerms[i][j];
    }
    //cout<<queryAvg[0]<<" "<<counter<<endl;
    for(int i = 0 ; i < queryAvg.size() ;i++)
        queryAvg[i] /= counter;
    //cout<<queryAvg[0]<<endl;

    //////////////////////////////////////////
    counter=0;
    TermInfoList *docTermInfoList =  ind.termInfoList(docID);
    docTermInfoList->startIteration();
    while(docTermInfoList->hasMore())
    {
        counter += 1;
        TermInfo *ti = docTermInfoList->nextEntry();
        if(wordEmbedding.find(ti->termID()) != wordEmbedding.end())
            docTerms.push_back(wordEmbedding[(ti->termID())]);
        else
            continue;

        //delete ti;
    }
    delete docTermInfoList;


    vector<double>docAvg (W2VecDimSize);//FIXME!!!!!!!!!!!!!!!!!!!!
    for(int i =0 ; i< docTerms.size() ; i++)
    {
        for(int j = 0 ;j < docTerms[i].size() ; j++)
            docAvg[j]+=docTerms[i][j];
    }
    //cout<<docAvg[0]<<" "<<counter<<endl;
    for(int i = 0 ; i < docAvg.size() ;i++)
        docAvg[i] /= counter;
    //cout<<docAvg[0]<<endl;


    return cosineSim(queryAvg , docAvg);
#endif

#if 0
    IndexedRealVector nonRel,rel;
    for (int i =0 ; i<nonReljudgDoc.size() ; i++)
    {
        nonRel.PushValue(nonReljudgDoc[i],0);
    }
    PseudoFBDocs  *nonRelDocs;
    nonRelDocs= new PseudoFBDocs(nonRel,-1,true);

    for(int i =0 ; i< relJudgDoc.size();i++)
        rel.PushValue(relJudgDoc[i],0);
    PseudoFBDocs  *relDocs;
    relDocs= new PseudoFBDocs(rel,-1,true);

    const QueryModel *qm = dynamic_cast<const QueryModel *>(textQR);
    //cout<<"positive score"<<endl;
    double sc = 0;
    DocumentRep *dRep;
    HashFreqVector hfv(ind,docID);

    dRep = computeDocRep(docID);
    textQR->startIteration();
    while (textQR->hasMore())
    {
        QueryTerm *qTerm = textQR->nextTerm();
        if(qTerm->id()==0)
        {
            cerr<<"**********"<<endl;
            //break;
            continue;
        }

        int tf;
        hfv.find(qTerm->id(),tf);
        DocInfo *info = new DocInfo(docID,tf);


        sc += scoreFunc()->matchedTermWeight(qTerm, textQR, info, dRep);//QL = sc+=|q|*log( p_seen(w|d)/(a(d)*p(w|C)) ) [slide7-11]
        //cout<<ind.term(qTerm->id())<<": "<<scoreFunc()->matchedTermWeight(qTerm, textQR, info, dRep)<<endl;
        delete info;
        delete qTerm;
    }

    double negQueryGenerationScore=0.0;
    //cout<<"negative score:"<<endl;
    if(RSMethodHM == 1)//RecSys(neg,coll)
    {
        negQueryGenerationScore= qm->negativeQueryGeneration(dRep ,nonReljudgDoc ,relJudgDoc,negGenModeHM, newNonRel,newRel,NegMu,delta,lambda_1,lambda_2);
    }
    else if (RSMethodHM == 2 || RSMethodHM == 3)//RecSys negKLQTE(2) and negKL(3)
    {
        negQueryGenerationScore = qm->negativeKL(dRep ,nonReljudgDoc , newNonRel,NegMu);
    }
    /*else if (RSMethodHM == 4)//fang
            {
                negQueryGenerationScore = fangScore(*nonRelDocs,docID,newNonRel);
            //    cout<<"inja5"<<endl;
            }*/

    //double fangScoreTmp = fangScore(*relDocs,docID,newRel);//considering positive feedback
    //negQueryGenerationScore -= fangScore(*relDocs,docID,newRel);//considering positive feedback


    double adjustedScore = scoreFunc()->adjustedScore(sc, textQR, dRep);
    //cout<<"negqueryScore: "<<negQueryGenerationScore<<endl;

    //cout<<"fangScoreTmp: "<< fangScoreTmp<<" negqueryScore: "<<negQueryGenerationScore<<" adjusted: "<<adjustedScore<<" newRel" <<newRel<<endl;
    //negQueryGenerationScore -= fangScoreTmp;

    //cout <<"inja6"<<endl;
    delete dRep;
    delete nonRelDocs;
    delete relDocs;
    // cout<<"neg score:**********:"<<negQueryGenerationScore<<endl;
    //cout<<"pos score:**********:"<<adjustedScore<<endl;
    return (negQueryGenerationScore + adjustedScore);

#endif

}


void lemur::retrieval::RetMethod::updateTextQuery(TextQueryRep &origRep,
                                                  const DocIDSet &relDocs,const DocIDSet &nonRelDocs )
{
    //cerr<<"fffffffffff"<<endl;
    QueryModel *qr;

    qr = dynamic_cast<QueryModel *> (&origRep);

    if(RM=="RM1"){
        computeRM1FBModel(*qr, relDocs,nonRelDocs);
        return;
    }else if(RM=="RM2"){
        computeRM2FBModel(*qr, relDocs);
        return;
    }else if(RM=="RM3"){
        computeRM3FBModel(*qr, relDocs);
        return;
    }else if(RM=="RM4"){
        computeRM4FBModel(*qr, relDocs);
        return;
    }else if(RM=="MIX"){
        computeMixtureFBModel(*qr, relDocs,nonRelDocs);
        return;
    }else if(RM=="DIVMIN"){
        computeDivMinFBModel(*qr, relDocs);
        return;
    }else if(RM=="MEDMM"){
        computeMEDMMFBModel(*qr, relDocs);
        return;
    }




    switch (qryParam.fbMethod) {
    case RetParameter::MIXTURE:
        computeMixtureFBModel(*qr, relDocs,nonRelDocs);
        break;
    case RetParameter::DIVMIN:
        computeDivMinFBModel(*qr, relDocs);
        break;
    case RetParameter::MARKOVCHAIN:
        computeMarkovChainFBModel(*qr, relDocs);
        break;
    case RetParameter::RM1:
        computeRM1FBModel(*qr, relDocs,nonRelDocs);
        break;
    case RetParameter::RM2:
        computeRM2FBModel(*qr, relDocs);
        break;
    default:
        throw Exception("SimpleKLRetMethod", "unknown feedback method");
        break;
    }
}

void lemur::retrieval::RetMethod::computeRelNonRelDist(TextQueryRep &origRep,
                                                       const vector<int> relDocs, const vector<int> nonRelDocs,bool isRelevant)
{

    cout<<"ddddddddddddddd";
    COUNT_T numTerms = ind.termCountUnique();

    lemur::langmod::DocUnigramCounter *dCounter;
    if(isRelevant)
        dCounter  = new lemur::langmod::DocUnigramCounter(relDocs, ind);
    else
        dCounter = new lemur::langmod::DocUnigramCounter(nonRelDocs, ind);


    double *distQuery = new double[numTerms+1];
    double *distQueryEst = new double[numTerms+1];


    double meanLL=1e-40;
    double distQueryNorm=0;

    for (int i=1; i<=numTerms;i++)
    {
        distQueryEst[i] = rand()+0.001;
        distQueryNorm += distQueryEst[i];
    }

    double noisePr = 0.9; //qryParam.fbMixtureNoise;
    int itNum = qryParam.emIterations;
    do {
        // re-estimate & compute likelihood
        double ll = 0;

        for (int i=1; i<=numTerms;i++)
        {
            distQuery[i] = distQueryEst[i]/distQueryNorm;
            // cerr << "dist: "<< distQuery[i] << endl;
            distQueryEst[i] =0;
        }

        distQueryNorm = 0;

        // compute likelihood
        dCounter->startIteration();
        while (dCounter->hasMore())
        {
            int wd; //dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);
            ll += wdCt * log (noisePr*collectLM->prob(wd)  // Pc(w)
                              + (1-noisePr)*distQuery[wd]); // Pq(w)
        }
        meanLL = 0.5*meanLL + 0.5*ll;
        if (fabs((meanLL-ll)/meanLL)< 0.0001)
        {
            //cerr << "converged at "<< qryParam.emIterations - itNum+1  << " with likelihood= "<< ll << endl;
            break;
        }

        // update counts
        dCounter->startIteration();
        while (dCounter->hasMore())
        {
            int wd; // dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);

            double prTopic = (1-noisePr)*distQuery[wd]/
                    ((1-noisePr)*distQuery[wd]+noisePr*collectLM->prob(wd));

            double incVal = wdCt*prTopic;
            distQueryEst[wd] += incVal;
            distQueryNorm += incVal;
        }
    } while (itNum-- > 0);

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (int i=1; i<=numTerms; i++)
        if (distQuery[i] > 0)
            lmCounter.incCount(i, distQuery[i]);


    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    //origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
    //                      qryParam.fbPrSumTh, qryParam.fbPrTh);

    vector<pair<double, int> >probWordVec;

    fblm->startIteration();
    while(fblm->hasMore())
    {
        int wid=-10;
        double wprob=0.0;
        fblm->nextWordProb(wid,wprob);
        probWordVec.push_back(pair<double,int>(wprob,wid));
    }

    ofstream outputfile;
    if(isRelevant)
        outputfile.open("outputfiles/mixtureRelDocs.txt",ios::app);
    else
        outputfile.open("outputfiles/mixtureNonRelDocs.txt",ios::app);

    set<int> queryWords;
    origRep.startIteration();
    while(origRep.hasMore())
        queryWords.insert(origRep.nextTerm()->id());

    vector<double> Vq(W2VecDimSize,0.0),Vwn(W2VecDimSize,0.0),Vbwn(W2VecDimSize,0.0);

    //cerr<<"here1111\n";

    std::sort(probWordVec.begin(),probWordVec.end(),pairCompare);
    if(probWordVec.size() > 0)
    {
        for(int i=probWordVec.size()-1 ; i> probWordVec.size()- 20 ; i--)
        {
            //cerr<<"here22222\n";
            //outputfile <<" ("<< ind.term(probWordVec[i].second)<<","<<probWordVec[i].first<<")";

            //cerr<<"here3333\n";
            if(isRelevant)
            {
                //cerr<<"here55555\n";
                for(int jj = 0 ;jj < W2VecDimSize ;jj++)
                    Vwn[jj] +=  wordEmbedding[probWordVec[i].second][jj] ;
            }
            else
            {
                if( queryWords.find(probWordVec[i].second) == queryWords.end() )//is not query word(ignore query words for Vwnb)
                {
                    //cerr<<"here6666\n";
                    for(int jj = 0 ;jj < W2VecDimSize ;jj++)
                        Vbwn[jj] += wordEmbedding[probWordVec[i].second][jj] ;
                }
            }

            //cerr<<"here77777\n";

        }
        if(isRelevant)
            for(int i=0 ;i < W2VecDimSize ; i++)
                Vwn[i]/=100.0;
        else
            for(int i=0 ;i < W2VecDimSize ; i++)
                Vbwn[i]/=100.0;

        //outputfile<<endl;
    }else
        cout<<"\nPROBWORDVEC size == 0 ????????\n\n";


    //}
    //outputfile<<endl;
    //}
    outputfile.close();


    cerr<<"here1111222222222\n";
    for(int i = 0 ; i < this->queryAvgVec.size() ; i++)
    {
        cout<<i<<" "<<queryAvgVec[i]<<" "<<Vwn[i]<<" "<<Vbwn[i]<<endl;
    }
    computeCoefMatrix(Vq , Vwn , Vbwn);

    delete fblm;
    delete dCounter;
    delete[] distQuery;
    delete[] distQueryEst;

}


void lemur::retrieval::RetMethod::multiplyVec2Vec(vector<double> m1, vector<double> m2, vector<vector<double> >&res )
{
    //cout<< endl<<W2VecDimSize<<endl;
    for(int i = 0 ;i < W2VecDimSize ;i++)
        for(int j = 0 ; j < W2VecDimSize ; j++)
            res[i][j] = m1[i] * m2[j];

}
void lemur::retrieval::RetMethod::multiplyMatrix2Vec(vector<vector<double> >m1 ,vector<double> m2,vector<double>&res  )
{
    cerr<<"hereaaaaaaaaaaa1111\n";


    //(N*N)^T * (1*N)^T
    for(int j = 0 ; j < W2VecDimSize ; j++)
        for(int k =0 ; k < W2VecDimSize ; k++)
        {
            //cerr<<"j "<<j <<" k "<<k<<" m1 "<<m1[k][j]<<" m2 "<<m2[k]<<endl;
            res[j] += m1[k][j] * m2[k];
        }
}
bool flag =true;
void lemur::retrieval::RetMethod::computeCoefMatrix(vector<double> Vq , vector<double> Vwn , vector<double> Vbwn)
{
//    cout<<"VQQQQQQQQQQQQQQQQQQQQQQQ\n";
//    for(int i = 0 ; i < Vq.size() ; i++)
//    {
//        cout<<i<<" "<<Vq[i]<<" "<<Vwn[i]<<" "<<Vbwn[i]<<endl;
//    }

    int epoch = 1;
    while(epoch-- && flag)
    {
        flag = false;
        vector<vector<double> >wnMatrix(W2VecDimSize ,vector<double>(W2VecDimSize , -10.0));
        for(int i = 0 ; i < W2VecDimSize ; i++)
            for(int j = 0 ; j < W2VecDimSize ; j++)
            {
                wnMatrix[i][j] = coefMatrix[i][j];

                //cout<< i <<" "<< j <<" ";
                //cout<<wnMatrix[i][j]<<endl;
            }

        vector<double>temp(W2VecDimSize ,-10.0);
        //rel
        for(int i = 0 ; i < Vwn.size() ; i++)
        {
            multiplyMatrix2Vec(wnMatrix,Vq,temp);
            cerr<<"here777777777777\n";
            for(int ii = 0 ; ii < temp.size(); ii++)
                temp[ii] = temp[ii] - Vwn[ii];
            multiplyVec2Vec(temp,Vq,wnMatrix);
            cerr<<"here88888888888\n";
        }

        break;/////////////////////////////////////////

        //nonRel
        vector<vector<double> >wnbMatrix(W2VecDimSize ,vector<double>(W2VecDimSize , -10.0));
        for(int i = 0 ; i < W2VecDimSize ; i++)
            for(int j = 0 ; j < W2VecDimSize ; j++)
                wnbMatrix[i][j] = coefMatrix[i][j];

        for(int i = 0 ; i < Vbwn.size() ; i++)
        {
            multiplyMatrix2Vec(wnbMatrix,Vq,temp);
            cerr<<"here99999999\n";
            for(int ii = 0 ; ii < temp.size(); ii++)
                temp[ii] = temp[ii] - Vbwn[ii];
            multiplyVec2Vec(temp,Vq,wnbMatrix);
            cerr<<"here10000000\n";
        }
        //diff
        for(int i = 0 ; i < wnbMatrix.size() ; i++)
            for( int j = 0 ; j < wnbMatrix[i].size() ; j++ )
                coefMatrix[i][j] = etaCoef*( alphaCoef*wnMatrix[i][j] - lambdaCoef*wnbMatrix[i][j] - betaCoef*coefMatrix[i][j]);


    }
}

void lemur::retrieval::RetMethod::computeMixtureFBModel(QueryModel &origRep,
                                                        const DocIDSet &relDocs, const DocIDSet &nonRelDocs )
{
    COUNT_T numTerms = ind.termCountUnique();

    lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    double *distQueryEst = new double[numTerms+1];

    double noisePr;

    int i;

    double meanLL=1e-40;
    double distQueryNorm=0;

    for (i=1; i<=numTerms;i++) {
        distQueryEst[i] = rand()+0.001;
        distQueryNorm+= distQueryEst[i];
    }
    noisePr = qryParam.fbMixtureNoise;

    int itNum = qryParam.emIterations;
    do {
        // re-estimate & compute likelihood
        double ll = 0;

        for (i=1; i<=numTerms;i++) {

            distQuery[i] = distQueryEst[i]/distQueryNorm;
            // cerr << "dist: "<< distQuery[i] << endl;
            distQueryEst[i] =0;
        }

        distQueryNorm = 0;

        // compute likelihood
        dCounter->startIteration();
        while (dCounter->hasMore()) {
            int wd; //dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);
            ll += wdCt * log (noisePr*collectLM->prob(wd)  // Pc(w)
                              + (1-noisePr)*distQuery[wd]); // Pq(w)
        }
        meanLL = 0.5*meanLL + 0.5*ll;
        if (fabs((meanLL-ll)/meanLL)< 0.0001) {
            cerr << "converged at "<< qryParam.emIterations - itNum+1
                 << " with likelihood= "<< ll << endl;
            break;
        }

        // update counts

        dCounter->startIteration();
        while (dCounter->hasMore()) {
            int wd; // dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);

            double prTopic = (1-noisePr)*distQuery[wd]/
                    ((1-noisePr)*distQuery[wd]+noisePr*collectLM->prob(wd));

            double incVal = wdCt*prTopic;
            distQueryEst[wd] += incVal;
            distQueryNorm += incVal;
        }
    } while (itNum-- > 0);

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
    delete dCounter;
    delete[] distQuery;
    delete[] distQueryEst;

}


void lemur::retrieval::RetMethod::computeDivMinFBModel(QueryModel &origRep,
                                                       const DocIDSet &relDocs)
{
    COUNT_T numTerms = ind.termCountUnique();

    double * ct = new double[numTerms+1];

    TERMID_T i;
    for (i=1; i<=numTerms; i++) ct[i]=0;

    COUNT_T actualDocCount=0;
    relDocs.startIteration();
    while (relDocs.hasMore()) {
        actualDocCount++;
        int id;
        double pr;
        relDocs.nextIDInfo(id,pr);
        DocModel *dm;
        dm = dynamic_cast<DocModel *> (computeDocRep(id));

        for (i=1; i<=numTerms; i++) { // pretend every word is unseen
            ct[i] += log(dm->unseenCoeff()*collectLM->prob(i));
        }

        TermInfoList *tList = ind.termInfoList(id);
        TermInfo *info;
        tList->startIteration();
        while (tList->hasMore()) {
            info = tList->nextEntry();
            ct[info->termID()] += log(dm->seenProb(info->count(), info->termID())/
                                      (dm->unseenCoeff()*collectLM->prob(info->termID())));
        }
        delete tList;
        delete dm;
    }
    if (actualDocCount==0) return;

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);

    double norm = 1.0/(double)actualDocCount;
    for (i=1; i<=numTerms; i++) {
        lmCounter.incCount(i,
                           exp((ct[i]*norm -
                                qryParam.fbMixtureNoise*log(collectLM->prob(i)))
                               / (1.0-qryParam.fbMixtureNoise)));
    }
    delete [] ct;


    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
}
void lemur::retrieval::RetMethod::computeMEDMMFBModel(QueryModel &origRep,
                                                      const DocIDSet &relDocs)
{
    // Write Your own MEDMM right here
}
void lemur::retrieval::RetMethod::computeMarkovChainFBModel(QueryModel &origRep, const DocIDSet &relDocs)
{
    int stopWordCutoff =50;

    lemur::utility::ArrayCounter<double> *counter = new lemur::utility::ArrayCounter<double>(ind.termCountUnique()+1);

    lemur::langmod::OneStepMarkovChain * mc = new lemur::langmod::OneStepMarkovChain(relDocs, ind, mcNorm,
                                                                                     1-qryParam.fbMixtureNoise);
    origRep.startIteration();
    double summ;
    while (origRep.hasMore()) {
        QueryTerm *qt;
        qt = origRep.nextTerm();
        summ =0;
        mc->startFromWordIteration(qt->id());
        // cout << " +++++++++ "<< ind.term(qt->id()) <<endl;
        TERMID_T fromWd;
        double fromWdPr;

        while (mc->hasMoreFromWord()) {
            mc->nextFromWordProb(fromWd, fromWdPr);
            if (fromWd <= stopWordCutoff) { // a stop word
                continue;
            }
            summ += qt->weight()*fromWdPr*collectLM->prob(fromWd);
            // summ += qt->weight()*fromWdPr;
        }
        if (summ==0) {
            // query term doesn't exist in the feedback documents, skip
            continue;
        }

        mc->startFromWordIteration(qt->id());
        while (mc->hasMoreFromWord()) {
            mc->nextFromWordProb(fromWd, fromWdPr);
            if (fromWd <= stopWordCutoff) { // a stop word
                continue;
            }

            counter->incCount(fromWd,
                              (qt->weight()*fromWdPr*collectLM->prob(fromWd)/summ));
            // counter->incCount(fromWd, (qt->weight()*fromWdPr/summ));

        }
        delete qt;
    }
    delete mc;

    lemur::langmod::UnigramLM *fbLM = new lemur::langmod::MLUnigramLM(*counter, ind.termLexiconID());

    origRep.interpolateWith(*fbLM, 1-qryParam.fbCoeff, qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fbLM;
    delete counter;
}

void lemur::retrieval::RetMethod::computeRM1FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs,const DocIDSet &nonRelDocs)
{
    COUNT_T numTerms = ind.termCountUnique();

    // RelDocUnigramCounter computes SUM(D){P(w|D)*P(D|Q)} for each w
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);
    lemur::langmod::RelDocUnigramCounter *nCounter = new lemur::langmod::RelDocUnigramCounter(nonRelDocs, ind);

    double *distQuery = new double[numTerms+1];
    double *negDistQuery = new double[numTerms+1];
    double expWeight = qryParam.fbCoeff;

    //double negWeight = 0.5;

    TERMID_T i;
    for (i=1; i<=numTerms;i++){
        distQuery[i] = 0.0;
        negDistQuery[i] = 0.0;
    }

    double pSum=0.0;
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf FIXME
        double wdPr;
        dCounter->nextCount(wd, wdPr);
        distQuery[wd]=wdPr;
        pSum += wdPr;
    }
    double nSum=0.0;
    nCounter->startIteration();
    while (nCounter->hasMore()) {
        int wd; // dmf FIXME
        double wdPr;
        nCounter->nextCount(wd, wdPr);
        negDistQuery[wd]=wdPr;
        nSum += wdPr;
    }


    for (i=1; i<=numTerms;i++) {
        //REMOVE  2 *
        if(feedbackMode == 2)
        {
            cout<<"normalFB"<<endl;
            distQuery[i] = expWeight*distQuery[i]/pSum +
                    (1-expWeight)*ind.termCount(i)/ind.termCount();

        }else if(feedbackMode == 1)
        {
            cout<<"ourFB"<<endl;
            distQuery[i] =  expWeight*(getNegWeight()*(distQuery[i]/pSum)-(1-getNegWeight())*(negDistQuery[i]/nSum) )+
                    (1-expWeight)*ind.termCount(i)/ind.termCount();
        }

        lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
        for (i=1; i<=numTerms; i++) {
            if (distQuery[i] > 0) {
                lmCounter.incCount(i, distQuery[i]);
            }
        }

        //cout<<"sum: "<<lmCounter.sum()<<endl;

        lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
        origRep.interpolateWith(*fblm, 0.0, qryParam.fbTermCount,
                                qryParam.fbPrSumTh, 0.0);
        delete fblm;
        delete dCounter;
        delete nCounter;
        delete[] distQuery;
        delete[] negDistQuery;
    }
}
void lemur::retrieval::RetMethod::computeRM3FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs)
{
    // Write Your own RM3 right here
}


// out: w.weight = P(w|Q)
// P(w|Q) = k P(w) P(Q|w)
// P(Q|w) = PROD_q P(q|w)
// P(q|w) = SUM_d P(q|d) P(w|d) p(d) / p(w)
// P(w) = SUM_d P(w|d) p(d)
// Promote this to some include somewhere...
struct termProb  {
    TERMID_T id; // TERM_ID
    double prob; // a*tf(w,d)/|d| +(1-a)*tf(w,C)/|C|
};

void lemur::retrieval::RetMethod::computeRM2FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs) {
    COUNT_T numTerms = ind.termCountUnique();
    COUNT_T termCount = ind.termCount();
    double expWeight = qryParam.fbCoeff;

    // RelDocUnigramCounter computes P(w)=SUM(D){P(w|D)*P(D|Q)} for each w
    // P(w) = SUM_d P(w|d) p(d)
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    COUNT_T numDocs = ind.docCount();
    vector<termProb> **tProbs = new vector<termProb> *[numDocs + 1];

    int i;
    for (i=1; i<=numTerms;i++)
        distQuery[i] = 0.0;
    for (i = 1; i <= numDocs; i++) {
        tProbs[i] = NULL;
    }

    // Put these in a faster structure.
    vector <TERMID_T> qTerms; // TERM_ID
    origRep.startIteration();
    while (origRep.hasMore()) {
        QueryTerm *qt = origRep.nextTerm();
        qTerms.push_back(qt->id());
        delete(qt);
    }
    COUNT_T numQTerms = qTerms.size();
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf fixme
        double P_w;
        double P_qw=0;
        double P_Q_w = 1.0;
        // P(q|w) = SUM_d P(q|d) P(w|d) p(d)
        dCounter->nextCount(wd, P_w);
        for (int j = 0; j < numQTerms; j++) {
            TERMID_T qtID = qTerms[j]; // TERM_ID
            relDocs.startIteration();
            while (relDocs.hasMore()) {
                int docID;
                double P_d, P_w_d, P_q_d;
                double dlength;
                relDocs.nextIDInfo(docID, P_d);
                dlength  = (double)ind.docLength(docID);
                if (tProbs[docID] == NULL) {
                    vector<termProb> * pList = new vector<termProb>;
                    TermInfoList *tList = ind.termInfoList(docID);
                    TermInfo *t;
                    tList->startIteration();
                    while (tList->hasMore()) {
                        t = tList->nextEntry();
                        termProb prob;
                        prob.id = t->termID();
                        prob.prob = expWeight*t->count()/dlength+
                                (1-expWeight)*ind.termCount(t->termID())/termCount;
                        pList->push_back(prob);
                    }
                    delete(tList);
                    tProbs[docID] = pList;
                }
                vector<termProb> * pList = tProbs[docID];
                P_w_d=0;
                P_q_d=0;
                for (int i = 0; i < pList->size(); i++) {
                    // p(q|d)= a*tf(q,d)/|d|+(1-a)*tf(q,C)/|C|
                    if((*pList)[i].id == qtID)
                        P_q_d = (*pList)[i].prob;

                    // p(w|d)= a*tf(w,d)/|d|+(1-a)*tf(w,C)/|C|
                    if((*pList)[i].id == wd)
                        P_w_d = (*pList)[i].prob;
                    if(P_q_d && P_w_d)
                        break;
                }
                P_qw += P_d*P_w_d*P_q_d;
            }
            // P(Q|w) = PROD_q P(q|w) / p(w)
            P_Q_w *= P_qw/P_w;
        }
        // P(w|Q) = k P(w) P(Q|w), k=1
        distQuery[wd] =P_w*P_Q_w;
    }

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, 0.0, qryParam.fbTermCount,
                            qryParam.fbPrSumTh, 0.0);
    delete fblm;
    delete dCounter;
    for (i = 1; i <= numDocs; i++) {
        delete(tProbs[i]);
    }
    delete[](tProbs);
    delete[] distQuery;
}

void lemur::retrieval::RetMethod::computeRM4FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs)
{
    cout<<"haha";
    // Write Your own RM4 right here
}

