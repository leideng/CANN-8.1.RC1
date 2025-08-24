#include <gtest/gtest.h>
#include <vector>
#include "add.h"

class AddTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "add test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "add test TearDown" << std::endl;
    }
};

TEST_F(AddTest, add_test_case_1)
{
    // define op
    ge::op::Add addOp;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    // update op input
    addOp.UpdateInputDesc("x1", tensorDesc);
    addOp.UpdateInputDesc("x2", tensorDesc);

    // call InferShapeAndType function
    auto ret = addOp.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // compare dtype and shape of op output
    auto outputDesc = addOp.GetOutputDescByName("y");
    EXPECT_EQ(outputDesc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expectedOutputShape = {2, 3, 4};
    EXPECT_EQ(outputDesc.GetShape().GetDims(), expectedOutputShape);
}
